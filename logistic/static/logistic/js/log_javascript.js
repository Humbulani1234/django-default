
/*----------------------------------------Background Color Changes---------------------------------------------------*/

document.addEventListener("DOMContentLoaded", function() {

    const colors = ["floralwhite", "white", "lightyellow"]; // Array of colors
    let currentIndex = 0; // Current index in the colors array

    function changeBackgroundColor() {

      document.body.style.backgroundColor = colors[currentIndex];
      currentIndex = (currentIndex + 1) % colors.length;

    }

    setInterval(changeBackgroundColor, 3000); // Change color every 3 seconds

});

/*---------------------------------------------------------------AJAX---------------------------------------------------*/

function getCookie(name) {
    
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(';').shift();
}

const csrftoken = getCookie('csrftoken');
const form = document.getElementById('probability-form');
const resetButton = document.getElementById('reset-button');
const probabilityResult = document.getElementById('probability-result');

    form.addEventListener('submit', function(event) {

      event.preventDefault();
      
      const formData = new FormData(form);
      
      // Send form data to the server using AJAX

      const headers = new Headers();
      headers.append('X-CSRFToken', csrftoken);

      fetch('http://localhost:8000/logistic/features/', {

        method: 'POST',
        body: formData,
        headers: headers
      })

      .then(response => response.json())
      .then(data => {
        probabilityResult.value = data.probability; // Use "value" to set input's value

      });

    });

    resetButton.addEventListener('click', function() {

      form.reset();
      probabilityResult.value = '';

});

/*-------------------------------------------------------Go back----------------------------------------------------------*/

function goBack() {

    window.history.back();
};

/*-----------------------------------------------------AJAX Simpler------------------------------------------------------

const form = document.getElementById('probability-form');
const resetButton = document.getElementById('reset-button');
const probabilityResult = document.getElementById('probability-result');

form.addEventListener('submit', async function(event) {

  event.preventDefault();

  const formData = new FormData(form);

  try {

    const response = await fetch('/calculate_probability/', {
      method: 'POST',
      body: formData,

    });

    const data = await response.json();
    probabilityResult.value = data.probability;
  }

  catch (error) {

    console.error('An error occurred:', error);
  }

});

resetButton.addEventListener('click', function() {

  form.reset();
  probabilityResult.value = '';

});     */
