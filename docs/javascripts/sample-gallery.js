(function () {
  "use strict";

  var activeTrigger = null;

  function ensureLightbox() {
    var existing = document.getElementById("ais-lightbox");
    if (existing) return existing;

    var dialog = document.createElement("dialog");
    dialog.id = "ais-lightbox";
    dialog.className = "ais-lightbox";
    dialog.setAttribute("aria-label", "Expanded sample image");
    dialog.innerHTML = [
      '<div class="ais-lightbox__bar">',
      '<button class="ais-lightbox__close" type="button" aria-label="Close expanded image">&#10005;</button>',
      "</div>",
      '<figure class="ais-lightbox__figure">',
      '<img class="ais-lightbox__image" alt="">',
      "</figure>",
      '<p class="ais-lightbox__caption" aria-live="polite"></p>'
    ].join("");
    document.body.appendChild(dialog);

    var closeButton = dialog.querySelector(".ais-lightbox__close");
    closeButton.addEventListener("click", function () {
      dialog.close();
    });
    dialog.addEventListener("click", function (event) {
      if (event.target === dialog) dialog.close();
    });
    dialog.addEventListener("cancel", function () {
      // Native dialog handles Escape. The close event restores focus below.
    });
    dialog.addEventListener("close", function () {
      document.body.classList.remove("ais-lightbox-open");
      if (activeTrigger && document.contains(activeTrigger)) activeTrigger.focus();
      activeTrigger = null;
    });
    return dialog;
  }

  function imageFor(trigger) {
    if (trigger.tagName === "IMG") return trigger;
    return trigger.querySelector("img");
  }

  function enhanceLightboxes(root) {
    root.querySelectorAll("[data-lightbox]:not([data-lightbox-ready])").forEach(function (trigger) {
      var source = imageFor(trigger);
      if (!source) return;

      trigger.dataset.lightboxReady = "true";
      if (trigger.tagName !== "A" && trigger.tagName !== "BUTTON") {
        trigger.setAttribute("role", "button");
        trigger.setAttribute("tabindex", "0");
      }
      trigger.setAttribute("aria-label", trigger.getAttribute("aria-label") || "Enlarge image: " + (source.alt || "sample"));

      function open(event) {
        event.preventDefault();
        var dialog = ensureLightbox();
        var target = dialog.querySelector(".ais-lightbox__image");
        var caption = dialog.querySelector(".ais-lightbox__caption");
        activeTrigger = trigger;
        target.src = trigger.getAttribute("href") || source.currentSrc || source.src;
        target.alt = source.alt || "Expanded sample";
        caption.textContent = trigger.dataset.caption || source.alt || "";
        document.body.classList.add("ais-lightbox-open");
        dialog.showModal();
        dialog.querySelector(".ais-lightbox__close").focus();
      }

      trigger.addEventListener("click", open);
      trigger.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") open(event);
      });
    });
  }

  function enhanceComparisons(root) {
    root.querySelectorAll(".image-compare:not([data-compare-ready])").forEach(function (figure, index) {
      var stage = figure.querySelector(".image-compare__stage");
      var after = figure.querySelector(".image-compare__after");
      if (!stage || !after) return;

      figure.dataset.compareReady = "true";
      var range = document.createElement("input");
      var focusRing = document.createElement("span");
      var divider = document.createElement("span");
      range.className = "image-compare__range";
      range.type = "range";
      range.min = "0";
      range.max = "100";
      range.value = figure.dataset.start || "50";
      range.setAttribute("aria-label", figure.dataset.label || "Reveal before and after images");
      range.setAttribute("aria-controls", figure.id || "image-compare-" + (index + 1));
      if (!figure.id) figure.id = "image-compare-" + (index + 1);
      focusRing.className = "image-compare__focus-ring";
      focusRing.setAttribute("aria-hidden", "true");
      divider.className = "image-compare__divider";
      divider.setAttribute("aria-hidden", "true");
      stage.appendChild(divider);
      stage.appendChild(range);
      stage.appendChild(focusRing);

      function update() {
        figure.style.setProperty("--compare-position", range.value + "%");
        range.setAttribute("aria-valuetext", range.value + "% after image revealed");
      }
      range.addEventListener("input", update);
      update();
    });
  }

  function init() {
    enhanceLightboxes(document);
    enhanceComparisons(document);
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(init);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
