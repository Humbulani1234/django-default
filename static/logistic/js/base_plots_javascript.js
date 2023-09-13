
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