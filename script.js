// ملف الجافاسكريبت الرئيسي 

// === Sword Slash Animation ===
document.addEventListener('DOMContentLoaded', function () {
  const sword = document.getElementById('sword-container');
  const slash = document.getElementById('slash-line');
  const audio = document.getElementById('sword-sound');
  const overlay = document.getElementById('slash-overlay');

  // Reset states
  sword.classList.remove('sword-animate');
  slash.classList.remove('slash-animate');
  if (overlay) overlay.classList.remove('hide');

  // Trigger animation
  setTimeout(() => {
    sword.classList.add('sword-animate');
    setTimeout(() => {
      slash.classList.add('slash-animate');
      if (audio) {
        audio.currentTime = 0;
        audio.play().catch(()=>{}); // Always try to play, ignore errors
      }
      // Hide sword and slash after animation
      setTimeout(() => {
        sword.classList.remove('sword-animate');
        slash.classList.remove('slash-animate');
        // Hide overlay to reveal the site
        if (overlay) overlay.classList.add('hide');
      }, 700);
    }, 220); // Slash timing after sword drop
  }, 300); // Initial delay for cinematic effect
});
// === End Sword Slash Animation === 