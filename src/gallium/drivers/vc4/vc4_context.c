/*
 * Copyright © 2014 Broadcom
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <xf86drm.h>
#include <err.h>

#include "pipe/p_defines.h"
#include "util/ralloc.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_blitter.h"
#include "util/u_upload_mgr.h"
#include "indices/u_primconvert.h"
#include "pipe/p_screen.h"

#include "vc4_screen.h"
#include "vc4_context.h"
#include "vc4_resource.h"

void
vc4_flush(struct pipe_context *pctx)
{
        vc4_flush_fd(pctx, -1, NULL);
}

void
vc4_flush_fd(struct pipe_context *pctx, int in_fence_fd, int *out_fence_fd)
{
        struct vc4_context *vc4 = vc4_context(pctx);

        /* vc4 is special in that it has no ring buffer to queue CLs.
         * So instead of submitting one big CL job come flush, we might have
         * many. Since there is only up to one of in/out fence fd each, we
         * submit the in fd on the first job and obtain the out fd for the
         * last one. The kernel guarantees jobs are executed in the order we
         * submit them since the seqno fences depend on that.
         */
        struct hash_entry *entry;
        hash_table_foreach(vc4->jobs, entry) {
                struct vc4_job *job = entry->data;
                bool needs_flush = job->needs_flush;

                /* Is this the last queued job? */
                if (_mesa_hash_table_next_entry(vc4->jobs, entry) == NULL) {
                        vc4_job_submit_fd(vc4, job, in_fence_fd, out_fence_fd);
                } else {
                        vc4_job_submit_fd(vc4, job, in_fence_fd, NULL);
                }

                /* Don't invoke the wait path in future submits. */
                if (needs_flush)
                        in_fence_fd = -1;
        }
}

static void
vc4_pipe_flush(struct pipe_context *pctx, struct pipe_fence_handle **fence,
               unsigned flags)
{
        struct vc4_context *vc4 = vc4_context(pctx);
        int out_fence_fd = -1;

        if (flags & PIPE_FLUSH_FENCE_FD) {
                vc4_flush_fd(pctx, vc4->in_fence_fd, &out_fence_fd);
        } else {
                vc4_flush_fd(pctx, vc4->in_fence_fd, NULL);
        }

        if (fence) {
                /* We need a valid fd even if nothing was flushed. */
                if ((flags & PIPE_FLUSH_FENCE_FD) && out_fence_fd == -1) {
                        /* If we haven't done any rendering at all that
                         * created an out fence, we ask the kernel to give
                         * us a fence fd for the last seqno. Otherwise we
                         * can reuse the last fence fd.
                         */
                        if (vc4->last_out_fence_fd == -1) {
                                vc4_get_seqno_fd(vc4, vc4->last_emit_seqno,
                                                 &out_fence_fd);
                                vc4->last_out_fence_fd = dup(out_fence_fd);
                        } else {
                                /* Ownership is transferred to the fence. */
                                out_fence_fd = dup(vc4->last_out_fence_fd);
                        }
                }

                struct pipe_screen *screen = pctx->screen;
                struct vc4_fence *f = vc4_fence_create(vc4->screen,
                                                       vc4->last_emit_seqno,
                                                       out_fence_fd);
                screen->fence_reference(screen, fence, NULL);
                *fence = (struct pipe_fence_handle *)f;
        }
}

/* We can't flush the texture cache within rendering a tile, so we have to
 * flush all rendering to the kernel so that the next job reading from the
 * tile gets a flushed cache.
 */
static void
vc4_texture_barrier(struct pipe_context *pctx, unsigned flags)
{
        vc4_flush(pctx);
}

static void
vc4_invalidate_resource(struct pipe_context *pctx, struct pipe_resource *prsc)
{
        struct vc4_context *vc4 = vc4_context(pctx);
        struct vc4_resource *rsc = vc4_resource(prsc);

        rsc->initialized_buffers = 0;

        struct hash_entry *entry = _mesa_hash_table_search(vc4->write_jobs,
                                                           prsc);
        if (!entry)
                return;

        struct vc4_job *job = entry->data;
        if (job->key.zsbuf && job->key.zsbuf->texture == prsc)
                job->resolve &= ~(PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL);
}

static void
vc4_context_destroy(struct pipe_context *pctx)
{
        struct vc4_context *vc4 = vc4_context(pctx);

        vc4_flush(pctx);

        if (vc4->blitter)
                util_blitter_destroy(vc4->blitter);

        if (vc4->primconvert)
                util_primconvert_destroy(vc4->primconvert);

        if (vc4->uploader)
                u_upload_destroy(vc4->uploader);

        slab_destroy_child(&vc4->transfer_pool);

        pipe_surface_reference(&vc4->framebuffer.cbufs[0], NULL);
        pipe_surface_reference(&vc4->framebuffer.zsbuf, NULL);

        vc4_program_fini(pctx);

        if (vc4->in_fence_fd >= 0)
                close(vc4->in_fence_fd);

        ralloc_free(vc4);
}

struct pipe_context *
vc4_context_create(struct pipe_screen *pscreen, void *priv, unsigned flags)
{
        struct vc4_screen *screen = vc4_screen(pscreen);
        struct vc4_context *vc4;

        /* Prevent dumping of the shaders built during context setup. */
        uint32_t saved_shaderdb_flag = vc4_debug & VC4_DEBUG_SHADERDB;
        vc4_debug &= ~VC4_DEBUG_SHADERDB;

        vc4 = rzalloc(NULL, struct vc4_context);
        if (!vc4)
                return NULL;
        struct pipe_context *pctx = &vc4->base;

        vc4->screen = screen;

        pctx->screen = pscreen;
        pctx->priv = priv;
        pctx->destroy = vc4_context_destroy;
        pctx->flush = vc4_pipe_flush;
        pctx->invalidate_resource = vc4_invalidate_resource;
        pctx->texture_barrier = vc4_texture_barrier;

        vc4_draw_init(pctx);
        vc4_state_init(pctx);
        vc4_program_init(pctx);
        vc4_query_init(pctx);
        vc4_resource_context_init(pctx);

        vc4_fence_context_init(vc4);
        vc4_job_init(vc4);

        vc4->fd = screen->fd;

        slab_create_child(&vc4->transfer_pool, &screen->transfer_pool);

	vc4->uploader = u_upload_create_default(&vc4->base);
	vc4->base.stream_uploader = vc4->uploader;
	vc4->base.const_uploader = vc4->uploader;

	vc4->blitter = util_blitter_create(pctx);
        if (!vc4->blitter)
                goto fail;

        vc4->primconvert = util_primconvert_create(pctx,
                                                   (1 << PIPE_PRIM_QUADS) - 1);
        if (!vc4->primconvert)
                goto fail;

        vc4_debug |= saved_shaderdb_flag;

        vc4->sample_mask = (1 << VC4_MAX_SAMPLES) - 1;

        return &vc4->base;

fail:
        pctx->destroy(pctx);
        return NULL;
}
