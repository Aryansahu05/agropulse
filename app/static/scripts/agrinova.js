(() => {
  function init() {
    const root = document.querySelector(".an-app");
    const sidebar = document.querySelector("[data-an-sidebar]");
    const toggle = document.querySelector("[data-an-toggle]");
    const overlay = document.querySelector("[data-an-overlay]");
    const page = document.querySelector("[data-an-page]");

    if (page) {
      // allow layout to paint first (smooth entry)
      requestAnimationFrame(() => page.classList.add("is-entered"));
    }

    const closeSidebar = () => {
      if (!root) return;
      root.classList.remove("is-sidebar-open");
    };

    const openSidebar = () => {
      if (!root) return;
      root.classList.add("is-sidebar-open");
    };

    if (toggle) {
      toggle.addEventListener("click", () => {
        if (!root) return;
        const isOpen = root.classList.contains("is-sidebar-open");
        if (isOpen) closeSidebar();
        else openSidebar();
      });
    }

    if (overlay) {
      overlay.addEventListener("click", closeSidebar);
    }

    // Close sidebar after clicking a nav link (mobile)
    if (sidebar) {
      sidebar.addEventListener("click", (e) => {
        const t = e.target;
        if (!(t instanceof Element)) return;
        const a = t.closest("a");
        if (!a) return;
        if (window.matchMedia && window.matchMedia("(max-width: 900px)").matches) {
          closeSidebar();
        }
      });
    }

    // Escape to close
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeSidebar();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();

