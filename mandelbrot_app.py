import threading
import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from tkinter import Tk, Canvas, NW, BOTH

import numpy as np
from PIL import Image, ImageTk
from matplotlib import cm

# Try to import Numba; gracefully fall back to pure NumPy
try:
    from numba import jit, cuda as numba_cuda
    HAS_NUMBA = True
    HAS_CUDA = numba_cuda.is_available()
except ImportError:
    HAS_NUMBA = False
    HAS_CUDA = False


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ComplexBounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def get_center(self) -> Tuple[float, float]:
        cx = (self.xmin + self.xmax) / 2
        cy = (self.ymin + self.ymax) / 2
        return cx, cy

    def get_width_height(self) -> Tuple[float, float]:
        w = self.xmax - self.xmin
        h = self.ymax - self.ymin
        return w, h

    def get_zoom_level(self) -> float:
        w, _ = self.get_width_height()
        initial_width = 3.0
        zoom = np.log2(initial_width / w) if w > 0 else 0
        return max(0, zoom)

    def zoom_to_rect(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        canvas_w: float,
        canvas_h: float,
    ) -> "ComplexBounds":
        xmin_pix = min(x0, x1)
        xmax_pix = max(x0, x1)
        ymin_pix = min(y0, y1)
        ymax_pix = max(y0, y1)

        xmin_new = self.xmin + (xmin_pix / canvas_w) * (self.xmax - self.xmin)
        xmax_new = self.xmin + (xmax_pix / canvas_w) * (self.xmax - self.xmin)
        ymin_new = self.ymin + (ymin_pix / canvas_h) * (self.ymax - self.ymin)
        ymax_new = self.ymin + (ymax_pix / canvas_h) * (self.ymax - self.ymin)

        return ComplexBounds(xmin_new, xmax_new, ymin_new, ymax_new)

    def zoom_out(self, cx: float, cy: float, scale: float = 2.0) -> "ComplexBounds":
        w, h = self.get_width_height()
        half_w = w * 0.5 * scale
        half_h = h * 0.5 * scale
        return ComplexBounds(cx - half_w, cx + half_w, cy - half_h, cy + half_h)

    def adjust_to_aspect_ratio(
        self, canvas_w: float, canvas_h: float
    ) -> "ComplexBounds":
        cx, cy = self.get_center()
        w, h = self.get_width_height()

        canvas_aspect = canvas_w / canvas_h if canvas_h > 0 else 1.0
        current_aspect = w / h if h > 0 else 1.0

        if canvas_aspect > current_aspect:
            new_w = h * canvas_aspect
            new_h = h
        else:
            new_w = w
            new_h = w / canvas_aspect

        new_xmin = cx - new_w / 2
        new_xmax = cx + new_w / 2
        new_ymin = cy - new_h / 2
        new_ymax = cy + new_h / 2

        return ComplexBounds(new_xmin, new_xmax, new_ymin, new_ymax)


# ============================================================================
# ADAPTIVE ITERATION & RESOLUTION
# ============================================================================

def calculate_adaptive_iterations(zoom_level: float) -> int:
    base_iterations = 100
    slope = 30
    max_iters = int(base_iterations + slope * zoom_level)
    return max(50, min(max_iters, 5000))


def calculate_adaptive_resolution(
    zoom_level: float, base_width: int, base_height: int
) -> Tuple[int, int]:
    if zoom_level < 50:
        scale = 1.0
    elif zoom_level < 100:
        scale = 0.7
    else:
        scale = 0.5

    width = max(200, int(base_width * scale))
    height = max(150, int(base_height * scale))
    return width, height


# ============================================================================
# COMPUTATION - NUMPY / NUMBA / CUDA
# ============================================================================

def mandelbrot_numpy(
    bounds: ComplexBounds,
    nx: int = 800,
    ny: int = 600,
    max_iterations: int = 100,
    smooth_coloring: bool = True,
) -> np.ndarray:
    x = np.linspace(bounds.xmin, bounds.xmax, nx)
    y = np.linspace(bounds.ymin, bounds.ymax, ny)
    X, Y = np.meshgrid(x, y)
    c = X + 1j * Y

    z = np.zeros_like(c, dtype=np.complex128)
    counts = np.zeros(c.shape, dtype=np.float32)
    escaped = np.zeros(c.shape, dtype=bool)
    bound_sq = 4.0

    for i in range(1, max_iterations + 1):
        not_escaped = ~escaped
        if not np.any(not_escaped):
            break

        z[not_escaped] = z[not_escaped] ** 2 + c[not_escaped]
        mag_sq = z.real * z.real + z.imag * z.imag
        newly_escaped = (mag_sq >= bound_sq) & (~escaped)

        if smooth_coloring and np.any(newly_escaped):
            mag = np.sqrt(mag_sq[newly_escaped])
            smooth_iter = i + 1 - np.log(np.log(mag)) / np.log(2)
            counts[newly_escaped] = smooth_iter
        else:
            counts[newly_escaped] = i

        escaped |= newly_escaped

    return counts


if HAS_NUMBA:
    @jit(nopython=True, parallel=True, cache=True)
    def mandelbrot_numba(
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        nx: int,
        ny: int,
        max_iterations: int,
        smooth_coloring: bool = True,
    ) -> np.ndarray:
        counts = np.zeros((ny, nx), dtype=np.float32)
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny
        log2 = np.log(2.0)

        for py in range(ny):
            y_val = ymin + py * dy
            for px in range(nx):
                x_val = xmin + px * dx

                z_real = 0.0
                z_imag = 0.0
                c_real = x_val
                c_imag = y_val

                for i in range(1, max_iterations + 1):
                    z_real_sq = z_real * z_real
                    z_imag_sq = z_imag * z_imag
                    mag_sq = z_real_sq + z_imag_sq

                    if mag_sq >= 4.0:
                        if smooth_coloring:
                            mag = np.sqrt(mag_sq)
                            smooth_iter = i + 1 - np.log(np.log(mag)) / log2
                            counts[py, px] = smooth_iter
                        else:
                            counts[py, px] = float(i)
                        break

                    z_imag = 2.0 * z_real * z_imag + c_imag
                    z_real = z_real_sq - z_imag_sq + c_real
                else:
                    counts[py, px] = float(max_iterations)

        return counts


if HAS_CUDA:
    @numba_cuda.jit
    def mandelbrot_cuda(
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        nx: int,
        ny: int,
        max_iterations: int,
        counts: np.ndarray,
    ) -> None:
        # px is x (column), py is y (row); match counts[py, px] [web:19]
        px, py = numba_cuda.grid(2)

        if px < nx and py < ny:
            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny
            log2 = math.log(2.0)

            x_val = xmin + px * dx
            y_val = ymin + py * dy

            z_real = 0.0
            z_imag = 0.0
            c_real = x_val
            c_imag = y_val

            for i in range(1, max_iterations + 1):
                z_real_sq = z_real * z_real
                z_imag_sq = z_imag * z_imag
                mag_sq = z_real_sq + z_imag_sq

                if mag_sq >= 4.0:
                    mag = math.sqrt(mag_sq)
                    smooth_iter = i + 1 - math.log(math.log(mag)) / log2
                    counts[py, px] = smooth_iter
                    return

                z_imag = 2.0 * z_real * z_imag + c_imag
                z_real = z_real_sq - z_imag_sq + c_real

            counts[py, px] = float(max_iterations)


def mandelbrot_cuda_wrapper(
    bounds: ComplexBounds,
    nx: int,
    ny: int,
    max_iterations: int,
) -> np.ndarray:
    # counts is (ny, nx) = (height, width)
    counts = np.zeros((ny, nx), dtype=np.float32)

    blockdim = (16, 16)  # (threads_x, threads_y)
    blocks_x = (nx + blockdim[0] - 1) // blockdim[0]
    blocks_y = (ny + blockdim[1] - 1) // blockdim[1]
    griddim = (blocks_x, blocks_y)

    d_counts = numba_cuda.to_device(counts)

    mandelbrot_cuda[griddim, blockdim](
        bounds.xmin, bounds.xmax,
        bounds.ymin, bounds.ymax,
        nx, ny, max_iterations, d_counts
    )

    d_counts.copy_to_host(counts)
    return counts


def mandelbrot(
    bounds: ComplexBounds,
    nx: int = 800,
    ny: int = 600,
    max_iterations: int = 100,
    use_gpu: bool = False,
    smooth_coloring: bool = True,
) -> np.ndarray:
    if use_gpu and HAS_CUDA:
        return mandelbrot_cuda_wrapper(bounds, nx, ny, max_iterations)
    elif HAS_NUMBA:
        return mandelbrot_numba(
            bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax,
            nx, ny, max_iterations, smooth_coloring
        )
    else:
        return mandelbrot_numpy(bounds, nx, ny, max_iterations, smooth_coloring)


def render_image(
    counts: np.ndarray,
    colormap_name: str = "hot",
    target_size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    norm = counts.astype(np.float32)
    maxc = norm.max() if norm.max() > 0 else 1.0
    norm = np.clip(norm / maxc, 0.0, 1.0)

    cmap = cm.get_cmap(colormap_name)
    rgba = (cmap(norm) * 255).astype(np.uint8)

    img = Image.fromarray(rgba, mode="RGBA")

    if target_size and img.size != target_size:
        img = img.resize(target_size, resample=Image.BICUBIC)

    return img


# ============================================================================
# GUI APPLICATION
# ============================================================================

class MandelApp:
    MAX_RENDER_WIDTH = 2560
    MAX_RENDER_HEIGHT = 1440
    DEFAULT_COLORMAP = "hot"
    SMOOTH_COLORING = True
    ENABLE_ADAPTIVE_RESOLUTION = True

    def __init__(self, width: int = 1000, height: int = 700, use_gpu: bool = False):
        self.root = Tk()
        self.root.title("Mandelbrot Zoom Viewer (CUDA‑fixed)")
        self.root.geometry(f"{width}x{height}")

        self.width = width
        self.height = height
        self.use_gpu = use_gpu and HAS_CUDA

        self.canvas = Canvas(
            self.root, width=self.width, height=self.height, bg="black"
        )
        self.canvas.pack(fill=BOTH, expand=True)

        self.bounds = ComplexBounds(xmin=-2.0, xmax=1.0, ymin=-1.2, ymax=1.2)
        self.bounds = self.bounds.adjust_to_aspect_ratio(self.width, self.height)

        self.colormap = self.DEFAULT_COLORMAP
        self.last_detailed_zoom = 0.0

        self._drag_start: Optional[Tuple[int, int]] = None
        self._rect_id: Optional[int] = None
        self.image_id: Optional[int] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None

        self._compute_lock = threading.Lock()
        self._render_thread: Optional[threading.Thread] = None
        self._should_stop = False
        self._resize_after_id: Optional[str] = None

        self.last_render_time = 0.0
        self.last_iterations = 0
        self.last_render_resolution = (0, 0)

        self._bind_events()

        backend = (
            "CUDA GPU"
            if self.use_gpu
            else ("Numba JIT" if HAS_NUMBA else "Pure NumPy")
        )
        features = []
        if self.SMOOTH_COLORING:
            features.append("smooth coloring")
        if self.ENABLE_ADAPTIVE_RESOLUTION:
            features.append("adaptive resolution")
        feature_str = f" ({', '.join(features)})" if features else ""
        print(f"Using backend: {backend}{feature_str}")

        self.render()

    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.root.bind("<Configure>", self.on_resize)
        self.root.bind("<MouseWheel>", self.on_scroll)
        self.root.bind("<Button-4>", self.on_scroll)
        self.root.bind("<Button-5>", self.on_scroll)
        self.root.bind("<r>", self.on_reset)
        self.root.bind("<q>", self.on_quit)
        # Optional: toggle GPU with 'g'
        self.root.bind("<g>", self.on_toggle_gpu)

    def on_toggle_gpu(self, event=None) -> None:
        if not HAS_CUDA:
            print("CUDA not available")
            return
        self.use_gpu = not self.use_gpu
        print("GPU enabled" if self.use_gpu else "GPU disabled")
        self.render()

    def on_resize(self, event: object) -> None:
        w = max(10, event.width)
        h = max(10, event.height)
        if w == self.width and h == self.height:
            return

        self.width = w
        self.height = h
        self.canvas.config(width=self.width, height=self.height)

        self.bounds = self.bounds.adjust_to_aspect_ratio(self.width, self.height)

        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(100, self.render)

    def on_button_press(self, event: object) -> None:
        self._drag_start = (event.x, event.y)
        if self._rect_id:
            self.canvas.delete(self._rect_id)
            self._rect_id = None

    def on_drag(self, event: object) -> None:
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y

        if self._rect_id:
            self.canvas.coords(self._rect_id, x0, y0, x1, y1)
        else:
            self._rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline="white", width=2
            )

    def on_button_release(self, event: object) -> None:
        if not self._drag_start:
            return

        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        self._drag_start = None

        if self._rect_id:
            self.canvas.delete(self._rect_id)
            self._rect_id = None

        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            return

        canvas_w = max(10, self.canvas.winfo_width())
        canvas_h = max(10, self.canvas.winfo_height())

        new_bounds = self.bounds.zoom_to_rect(
            x0, y0, x1, y1, canvas_w, canvas_h
        )
        self.bounds = new_bounds.adjust_to_aspect_ratio(canvas_w, canvas_h)
        self.render()

    def on_right_click(self, event: object) -> None:
        canvas_w = max(10, self.canvas.winfo_width())
        canvas_h = max(10, self.canvas.winfo_height())

        cx = self.bounds.xmin + (event.x / canvas_w) * (
            self.bounds.xmax - self.bounds.xmin
        )
        cy = self.bounds.ymin + (event.y / canvas_h) * (
            self.bounds.ymax - self.bounds.ymin
        )
        new_bounds = self.bounds.zoom_out(cx, cy, scale=2.0)
        self.bounds = new_bounds.adjust_to_aspect_ratio(canvas_w, canvas_h)
        self.render()

    def on_scroll(self, event: object) -> None:
        canvas_w = max(10, self.canvas.winfo_width())
        canvas_h = max(10, self.canvas.winfo_height())

        cx = self.bounds.xmin + (event.x / canvas_w) * (
            self.bounds.xmax - self.bounds.xmin
        )
        cy = self.bounds.ymin + (event.y / canvas_h) * (
            self.bounds.ymax - self.bounds.ymin
        )

        scale = 0.5 if getattr(event, "delta", 0) > 0 else 2.0
        if hasattr(event, "num"):
            scale = 0.5 if event.num == 4 else 2.0

        new_bounds = self.bounds.zoom_out(cx, cy, scale=scale)
        self.bounds = new_bounds.adjust_to_aspect_ratio(canvas_w, canvas_h)
        self.render()

    def on_reset(self, event: object) -> None:
        canvas_w = max(10, self.canvas.winfo_width())
        canvas_h = max(10, self.canvas.winfo_height())

        self.bounds = ComplexBounds(xmin=-2.0, xmax=1.0, ymin=-1.2, ymax=1.2)
        self.bounds = self.bounds.adjust_to_aspect_ratio(canvas_w, canvas_h)
        self.last_detailed_zoom = 0.0
        self.render()

    def on_quit(self, event: object) -> None:
        self._should_stop = True
        self.root.quit()

    def render(self) -> None:
        self._should_stop = False

        canvas_w = max(10, self.canvas.winfo_width())
        canvas_h = max(10, self.canvas.winfo_height())

        zoom_level = self.bounds.get_zoom_level()

        if self.ENABLE_ADAPTIVE_RESOLUTION:
            render_width, render_height = calculate_adaptive_resolution(
                zoom_level, canvas_w, canvas_h
            )
        else:
            render_width = min(canvas_w, self.MAX_RENDER_WIDTH)
            render_height = min(canvas_h, self.MAX_RENDER_HEIGHT)

        thread = threading.Thread(
            target=self._render_job,
            args=(render_width, render_height, zoom_level, canvas_w, canvas_h),
            daemon=True,
        )
        self._render_thread = thread
        thread.start()

    def _render_job(
        self,
        render_width: int,
        render_height: int,
        zoom_level: float,
        canvas_w: int,
        canvas_h: int,
    ) -> None:
        try:
            with self._compute_lock:
                start_time = time.time()

                max_iterations = calculate_adaptive_iterations(zoom_level)

                counts = mandelbrot(
                    bounds=self.bounds,
                    nx=render_width,
                    ny=render_height,
                    max_iterations=max_iterations,
                    use_gpu=self.use_gpu,
                    smooth_coloring=self.SMOOTH_COLORING,
                )

                img = render_image(
                    counts,
                    colormap_name=self.colormap,
                    target_size=(canvas_w, canvas_h),
                )

                photo = ImageTk.PhotoImage(img)
                elapsed = time.time() - start_time

                self.last_render_time = elapsed
                self.last_iterations = max_iterations
                self.last_render_resolution = (render_width, render_height)
                self.last_detailed_zoom = max(self.last_detailed_zoom, zoom_level)

                def update_ui() -> None:
                    if self._should_stop:
                        return
                    self.photo_image = photo
                    if self.image_id is None:
                        self.image_id = self.canvas.create_image(
                            0, 0, anchor=NW, image=self.photo_image
                        )
                    else:
                        self.canvas.itemconfig(self.image_id, image=self.photo_image)

                    zoom_level_now = self.bounds.get_zoom_level()
                    res_str = f"{self.last_render_resolution[0]}×{self.last_render_resolution[1]}"

                    if zoom_level_now > 50 and self.ENABLE_ADAPTIVE_RESOLUTION:
                        res_str += " (adaptive)"

                    self.root.title(
                        f"Mandelbrot | "
                        f"Zoom: {zoom_level_now:.1f}x | "
                        f"Iters: {self.last_iterations} | "
                        f"Res: {res_str} | "
                        f"Time: {elapsed:.2f}s | "
                        f"GPU: {self.use_gpu}"
                    )

                self.root.after(0, update_ui)

        except Exception as e:
            print(f"Render error: {e}")
            import traceback
            traceback.print_exc()

    def run(self) -> None:
        self.root.mainloop()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    use_gpu = HAS_CUDA

    if use_gpu:
        print("CUDA GPU available - using GPU acceleration")
    elif HAS_NUMBA:
        print("Numba JIT available - using CPU multi-threading")
    else:
        print("Numba not installed - using pure NumPy (slow)")
        print("Install with: pip install numba")

    app = MandelApp(1000, 700, use_gpu=use_gpu)
    app.run()
