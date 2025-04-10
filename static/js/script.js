document.querySelector("form").onsubmit = async function(event) {
  event.preventDefault();
  
  const formData = new FormData(event.target);
  
  const response = await fetch("/upload", {
    method: "POST",
    body: formData
  });

  const result = await response.json();
  document.getElementById("result").textContent = `Correlation: ${result.correlation}`;
  document.getElementById("downloadLink").style.display = "block";
};
