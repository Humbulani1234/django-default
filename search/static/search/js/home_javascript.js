
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

/*----------------------------------------------Search Field-------------------------------------------------*/

document.addEventListener("DOMContentLoaded", function() {
    const searchInput = document.getElementById("search-input");
    const datalist = document.getElementById("suggestions");
    const searchButton = document.getElementById("search-button");

    searchInput.addEventListener("input", function() {
        datalist.innerHTML = "";  // Clear suggestions when typing
    });

searchButton.addEventListener("click", function() {

    const selectedName = searchInput.value;
    if (selectedName == "logistic") {
        const url = `http://localhost:8000/${encodeURIComponent(selectedName)}/features`;
        window.location.href = url;}
    else if (selectedName == "decision") {
        const url = `http://localhost:8000/${encodeURIComponent(selectedName)}/decisionfeatures`;
        window.location.href = url;} 
    else if (selectedName == "streamlit") {
        const url = `http://localhost:8000/search/streamlit`;
        window.location.href = url;}
    else if (selectedName == "github-pd") {
        const url = `http://localhost:8000/search/github-django-pd/`;
        window.location.href = url;}
    else if (selectedName == "github-streamlit") {
        const url = `http://localhost:8000/search/github-django-streamlit/`;
        window.location.href = url;}  
    else if (selectedName == "streamlit") {
        const url = `http://localhost:8000/search/streamlit/`;
        window.location.href = url;}  
    else if (selectedName == "real-analysis") {
        const url = `http://localhost:8000/search/real-analysis/`;
        window.location.href = url;}                                                
});

searchInput.addEventListener("input", async function() {
    
    const query = this.value;
    if (query.length >= 1) {
        try {
            const response = await fetch(`http://localhost:8000/?q=${query}`);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();

            datalist.innerHTML = "";
            data.results.forEach(function(item) {
                const option = document.createElement("option");
                option.value = item.menu;
                datalist.appendChild(option);
            });
        } catch (error) {
            console.error("Error fetching data: ", error);
        }
    } else {
        datalist.innerHTML = "";
    }
});
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

