
/*----------------------------------------------------Text Animation-----------------------------------------------*/

document.addEventListener("DOMContentLoaded", function() {

  const section1Element = document.querySelector(".e");
  const section2Element = document.querySelector(".f");

  const section1Text = "Choose Model.";
  const section2Text = "Github Repos.";

  function typeText(activeElement, text) {

    let index = 0;

    function updateText() {

      activeElement.textContent = text.slice(0, index);

      index++;
      if (index > text.length) {
        clearInterval(typingInterval);

      }

    }

    const typingInterval = setInterval(updateText, 100); // Adjust interval for typing speed
  }

  function startTyping() {

    typeText(section1Element, section1Text);

    setTimeout(function() {

      typeText(section2Element, section2Text);
    }, section1Text.length * 100 + 1000); // Start second section after a delay

    setTimeout(startTyping, 5000); // Start typing animation again after 5 seconds
  }

  startTyping();

});

/*----------------------------------------Background Color Changes---------------------------------------------------*/

document.addEventListener("DOMContentLoaded", function() {

    const colors = ["floralwhite", "white", "lightyellow"]; // Array of colors
    let currentIndex = 0; // Current index in the colors array

    function changeBackgroundColor() {

      document.body.style.backgroundColor = colors[currentIndex];
      currentIndex = (currentIndex + 1) % colors.length;
      
    }

    setInterval(changeBackgroundColor, 3000); // Change color every 5 seconds

});

/*----------------------------------------Page Loading---------------------------------------------------*/

// document.addEventListener('DOMContentLoaded', function() {

//     setTimeout(function() {
//     document.getElementById('loading-message').style.display = 'none'
//     }, 100000);
// });

// document.addEventListener('load', function() {
//     setTimeout(function() {
//     document.getElementById('loading-message').style.display = 'none'
//     }, 100000)
// });

