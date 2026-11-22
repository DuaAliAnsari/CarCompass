(function () {
  const stage = document.querySelector(".hero-stage");
  const video = document.querySelector(".hero-video");
  const slides = Array.from(document.querySelectorAll(".hero-slide"));

  function startSlides() {
    if (slides.length < 2) return;
    let index = 0;
    setInterval(() => {
      slides[index].classList.remove("is-on");
      index = (index + 1) % slides.length;
      slides[index].classList.add("is-on");
    }, 5500);
  }

  if (!video) {
    startSlides();
    return;
  }

  const markReady = () => {
    if (stage) stage.classList.add("has-video");
  };

  video.addEventListener("loadeddata", markReady);
  video.addEventListener("playing", markReady);
  video.addEventListener("error", startSlides);

  const play = video.play();
  if (play && typeof play.then === "function") {
    play.then(markReady).catch(startSlides);
  } else if (video.readyState >= 2) {
    markReady();
  } else {
    startSlides();
  }
})();
