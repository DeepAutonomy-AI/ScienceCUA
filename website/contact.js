(() => {
  const trigger = document.getElementById("contact-trigger");
  const dialog = document.getElementById("contact-dialog");
  const status = document.getElementById("contact-status");
  const resetTimers = new Map();

  trigger.addEventListener("click", () => {
    dialog.showModal();
    trigger.setAttribute("aria-expanded", "true");
    document.documentElement.classList.add("contact-open");
  });

  dialog.querySelector(".contact-close").addEventListener("click", () => dialog.close());

  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });

  dialog.addEventListener("close", () => {
    trigger.setAttribute("aria-expanded", "false");
    document.documentElement.classList.remove("contact-open");
    status.textContent = "";
    trigger.focus({ preventScroll: true });
  });

  async function copyEmail(email) {
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(email);
        return;
      } catch {
        // Support browsers that restrict clipboard access or local-file previews.
      }
    }

    const field = document.createElement("textarea");
    const focusedElement = document.activeElement;
    field.value = email;
    field.readOnly = true;
    field.style.cssText = "position: fixed; opacity: 0; pointer-events: none;";
    dialog.append(field);

    try {
      field.select();
      if (!document.execCommand("copy")) throw new Error("Copy unavailable");
    } finally {
      field.remove();
      focusedElement?.focus({ preventScroll: true });
    }
  }

  dialog.querySelectorAll("[data-copy-email]").forEach((button) => {
    button.addEventListener("click", async () => {
      if (button.getAttribute("aria-busy") === "true") return;
      const email = button.dataset.copyEmail;
      const label = button.querySelector("span");
      button.setAttribute("aria-busy", "true");

      try {
        await copyEmail(email);
        clearTimeout(resetTimers.get(button));
        button.classList.add("is-copied");
        label.textContent = "Copied!";
        status.textContent = `Copied ${email} to clipboard.`;
        resetTimers.set(button, setTimeout(() => {
          button.classList.remove("is-copied");
          label.textContent = "Copy";
          if (status.textContent === `Copied ${email} to clipboard.`) status.textContent = "";
          resetTimers.delete(button);
        }, 2500));
      } catch {
        status.textContent = "Couldn’t copy automatically. Select the email address to copy it.";
      } finally {
        button.removeAttribute("aria-busy");
      }
    });
  });
})();
