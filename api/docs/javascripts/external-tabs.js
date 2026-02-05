document.addEventListener("DOMContentLoaded", () => {
  document
    .querySelectorAll(".md-tabs a[href^='http']")
    .forEach(link => {
      link.setAttribute("target", "_blank");
      link.setAttribute("rel", "noopener");
    });
});