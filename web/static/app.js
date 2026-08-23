// Progressive enhancement only: the dashboard works with JS disabled.
(function () {
  "use strict";

  // Repeating form blocks (experience, education, certifications, projects).
  function wireRepeaters() {
    document.querySelectorAll("[data-repeat-add]").forEach(function (button) {
      button.addEventListener("click", function () {
        var name = button.getAttribute("data-repeat-add");
        var container = document.querySelector('[data-repeat-container="' + name + '"]');
        var template = document.querySelector('[data-repeat-template="' + name + '"]');
        if (!container || !template) return;

        var index = container.querySelectorAll("[data-repeat-item]").length;
        var html = template.innerHTML.replace(/__INDEX__/g, String(index));
        var wrapper = document.createElement("div");
        wrapper.innerHTML = html;
        var node = wrapper.firstElementChild;
        container.appendChild(node);
        var first = node.querySelector("input, textarea, select");
        if (first) first.focus();
      });
    });

    document.addEventListener("click", function (event) {
      var target = event.target.closest("[data-repeat-remove]");
      if (!target) return;
      var item = target.closest("[data-repeat-item]");
      if (item) item.remove();
    });
  }

  // Auto-refresh pages that show live state while a run is in flight.
  function wireAutoRefresh() {
    var node = document.querySelector("[data-refresh-seconds]");
    if (!node) return;
    var seconds = parseInt(node.getAttribute("data-refresh-seconds"), 10);
    if (!seconds || seconds < 2) return;
    setTimeout(function () { window.location.reload(); }, seconds * 1000);
  }

  // Copy-to-clipboard for secrets shown once.
  function wireCopy() {
    document.querySelectorAll("[data-copy]").forEach(function (button) {
      button.addEventListener("click", function () {
        var value = button.getAttribute("data-copy");
        if (!navigator.clipboard) return;
        navigator.clipboard.writeText(value).then(function () {
          var original = button.textContent;
          button.textContent = "Copied";
          setTimeout(function () { button.textContent = original; }, 1500);
        });
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    wireRepeaters();
    wireAutoRefresh();
    wireCopy();
  });
})();
