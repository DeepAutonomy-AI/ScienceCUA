(() => {
  const player = document.getElementById("promo-video");
  const buttons = document.querySelectorAll(".promo-languages button");
  if (!player || !buttons.length) return;

  const playButton = document.getElementById("promo-play");
  playButton.hidden = false;
  playButton.addEventListener("click", () => {
    player.play().catch(() => {
      playButton.hidden = false;
    });
  });
  player.addEventListener("play", () => {
    playButton.hidden = true;
  });

  let switchId = 0;

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      if (button.getAttribute("aria-pressed") === "true") return;

      const currentSwitch = ++switchId;
      const position = player.currentTime;
      const resumePlayback = !player.paused && !player.ended;
      player.pause();

      buttons.forEach((option) => {
        option.setAttribute("aria-pressed", String(option === button));
      });
      player.poster = button.dataset.videoPoster;
      player.setAttribute("aria-label", button.dataset.videoLabel);
      playButton.querySelector("img").src = button.dataset.videoPoster;
      playButton.setAttribute("aria-label", `Play ${button.dataset.videoLabel}`);
      player.preload = "metadata";
      player.addEventListener("loadedmetadata", () => {
        if (currentSwitch !== switchId) return;
        if (Number.isFinite(player.duration)) {
          player.currentTime = Math.min(position, player.duration);
        }
        if (resumePlayback) {
          player.play().catch(() => {
            // Native controls remain available if the browser requires another click.
          });
        }
      }, { once: true });
      player.src = button.dataset.videoSrc;
      player.load();
    });
  });
})();
