// document.addEventListener("DOMContentLoaded", () => {
//   // --- SAFE INITIALIZATION ---
//   // These variables and listeners are now set up only after the page is fully loaded.
//   const themeToggle = document.getElementById("theme-toggle");
//   const body = document.body;

//   // Applies the saved theme from localStorage
//   function applySavedTheme() {
//     const savedTheme = localStorage.getItem("theme");
//     if (savedTheme === "light") {
//       body.classList.add("light-mode");
//     }
//   }

//   // Sets up the theme toggle button click event
//   if (themeToggle) {
//     themeToggle.addEventListener("click", () => {
//       body.classList.toggle("light-mode");
//       if (body.classList.contains("light-mode")) {
//         localStorage.setItem("theme", "light");
//       } else {
//         localStorage.setItem("theme", "dark");
//       }
//     });
//   }

//   // Sets up the glowing cursor mouse movement event
//   document.addEventListener("mousemove", (e) => {
//     const cursorLight = document.querySelector(".cursor-light");
//     if (cursorLight) {
//       cursorLight.style.left = e.clientX + "px";
//       cursorLight.style.top = e.clientY + "px";
//     }
//   });

//   // --- RUN INITIAL FUNCTIONS ---
//   applySavedTheme();
//   initializeFloatingImages();

//   // --- PAGE-SPECIFIC INITIALIZATIONS ---
//   if (document.querySelector(".slideshow-container")) {
//     generateDots();
//     showSlides(slideIndex);
//   }
//   if (document.querySelector(".chat-container")) {
//     initializeChatbot();
//   }
//   if (document.querySelector(".prediction-container")) {
//     initializePredictionPage();
//   }
// });

// // --- Slideshow Logic ---
// // These functions remain global so they can be called by HTML onclick attributes.
// let slideIndex = 1;
// let autoSlideTimeout;

// function generateDots() {
//   const slides = document.getElementsByClassName("slide-card");
//   const dotsContainer = document.querySelector(".dots-container");
//   if (!dotsContainer) return;
//   dotsContainer.innerHTML = "";
//   for (let i = 0; i < slides.length; i++) {
//     const dot = document.createElement("span");
//     dot.classList.add("dot");
//     dot.setAttribute("onclick", `currentSlide(${i + 1})`);
//     dotsContainer.appendChild(dot);
//   }
// }

// function plusSlides(n) {
//   clearTimeout(autoSlideTimeout);
//   showSlides((slideIndex += n));
// }

// function currentSlide(n) {
//   clearTimeout(autoSlideTimeout);
//   showSlides((slideIndex = n));
// }

// function showSlides(n) {
//   let i;
//   let slides = document.getElementsByClassName("slide-card");
//   let dots = document.getElementsByClassName("dot");
//   if (slides.length === 0 || dots.length === 0) return;

//   if (n > slides.length) {
//     slideIndex = 1;
//   }
//   if (n < 1) {
//     slideIndex = slides.length;
//   }

//   for (i = 0; i < slides.length; i++) {
//     slides[i].classList.remove("active-slide");
//   }
//   for (i = 0; i < dots.length; i++) {
//     dots[i].className = dots[i].className.replace(" active", "");
//   }

//   slides[slideIndex - 1].classList.add("active-slide");
//   dots[slideIndex - 1].className += " active";

//   autoSlideTimeout = setTimeout(() => plusSlides(1), 20000);
// }

// // --- FLOATING IMAGE ANIMATION LOGIC ---
// let images = [];

// function initializeFloatingImages() {
//   const imageElements = document.querySelectorAll(".floating-img");
//   imageElements.forEach((img) => {
//     images.push({
//       element: img,
//       x: Math.random() * (window.innerWidth - 150),
//       y: Math.random() * (window.innerHeight - 150),
//       dx: (Math.random() - 0.5) * 6,
//       dy: (Math.random() - 0.5) * 6,
//     });
//   });

//   if (images.length > 0) {
//     animate();
//   }
// }

// function animate() {
//   images.forEach((img, index) => {
//     img.x += img.dx;
//     img.y += img.dy;

//     if (img.x <= 0 || img.x >= window.innerWidth - 150) img.dx *= -1;
//     if (img.y <= 0 || img.y >= window.innerHeight - 150) img.dy *= -1;

//     for (let i = index + 1; i < images.length; i++) {
//       const otherImg = images[i];
//       const dist = Math.sqrt(
//         (img.x - otherImg.x) ** 2 + (img.y - otherImg.y) ** 2
//       );
//       if (dist < 150) {
//         [img.dx, otherImg.dx] = [otherImg.dx, img.dx];
//         [img.dy, otherImg.dy] = [otherImg.dy, img.dy];
//       }
//     }
//     img.element.style.transform = `translate(${img.x}px, ${img.y}px)`;
//   });

//   requestAnimationFrame(animate);
// }

// // --- CHATBOT LOGIC ---
// function initializeChatbot() {
//   const chatBox = document.getElementById("chat-box");
//   const userInput = document.getElementById("user-input");
//   const sendButton = document.getElementById("send-button");
//   const suggestionsContainer = document.getElementById("suggestion-cards");
//   const suggestions = [
//     "What is an Epidural Hemorrhage?",
//     "Tell me about Subdural Hemorrhage.",
//     "Explain Subarachnoid Hemorrhage.",
//     "What are the signs of IPH?",
//   ];

//   suggestions.forEach((text) => {
//     const card = document.createElement("div");
//     card.className = "suggestion-card";
//     card.textContent = text;
//     card.onclick = () => {
//       userInput.value = text;
//       sendMessage();
//     };
//     suggestionsContainer.appendChild(card);
//   });

//   addMessageToBox(
//     "Hello! I'm the TBiDx Assistant. How can I help you understand different types of Traumatic Brain Injuries today?",
//     "bot"
//   );

//   sendButton.addEventListener("click", sendMessage);
//   userInput.addEventListener("keypress", (e) => {
//     if (e.key === "Enter") sendMessage();
//   });

//   async function sendMessage() {
//     const messageText = userInput.value.trim();
//     if (!messageText) return;

//     addMessageToBox(messageText, "user");
//     userInput.value = "";
//     suggestionsContainer.style.display = "none";

//     try {
//       const response = await fetch("http://127.0.0.1:8000/summarise/", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ diagnosis: messageText }),
//       });

//       if (!response.ok)
//         throw new Error(`HTTP error! status: ${response.status}`);

//       const data = await response.json();
//       const botReply =
//         data.summary || data.error || "Sorry, I couldn't get a response.";
//       addMessageToBox(botReply, "bot");
//     } catch (error) {
//       console.error("Error fetching from API:", error);
//       addMessageToBox(
//         "I'm having trouble connecting to my knowledge base. Please try again later.",
//         "bot"
//       );
//     }
//   }

//   function addMessageToBox(text, sender) {
//     if (sender === "user") {
//       const messageElement = document.createElement("div");
//       messageElement.className = "chat-message user";
//       messageElement.textContent = text;
//       chatBox.appendChild(messageElement);
//     } else {
//       const botMessageContainer = document.createElement("div");
//       botMessageContainer.className = "bot-message-container";
//       const iconElement = document.createElement("div");
//       iconElement.className = "bot-icon";
//       iconElement.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect x="4" y="12" width="8" height="8" rx="2"/><path d="M8 12v-2a2 2 0 1 1 4 0v2"/></svg>`;
//       const messageElement = document.createElement("div");
//       messageElement.className = "chat-message bot";
//       messageElement.innerHTML = marked.parse(text);
//       botMessageContainer.appendChild(iconElement);
//       botMessageContainer.appendChild(messageElement);
//       chatBox.appendChild(botMessageContainer);
//     }
//     chatBox.scrollTop = chatBox.scrollHeight;
//   }
// }

// // --- PREDICTION PAGE LOGIC ---
// function initializePredictionPage() {
//   const uploadBox = document.getElementById("upload-box");
//   const imageUpload = document.getElementById("image-upload");
//   const imagePreviewContainer = document.getElementById(
//     "image-preview-container"
//   );
//   const imagePreview = document.getElementById("image-preview");
//   const resultsContainer = document.getElementById("results-container");
//   const summaryContainer = document.getElementById("summary-container");
//   const loader = document.getElementById("loader");

//   imageUpload.addEventListener("change", (e) => {
//     const file = e.target.files[0];
//     if (file) handleFile(file);
//   });

//   uploadBox.addEventListener("dragover", (e) => {
//     e.preventDefault();
//     uploadBox.classList.add("dragover");
//   });
//   uploadBox.addEventListener("dragleave", () => {
//     uploadBox.classList.remove("dragover");
//   });
//   uploadBox.addEventListener("drop", (e) => {
//     e.preventDefault();
//     uploadBox.classList.remove("dragover");
//     const file = e.dataTransfer.files[0];
//     if (file) handleFile(file);
//   });

//   function handleFile(file) {
//     const reader = new FileReader();
//     reader.onload = (e) => {
//       imagePreview.src = e.target.result;
//       imagePreviewContainer.style.display = "block";
//       uploadBox.style.display = "none";
//     };
//     reader.readAsDataURL(file);

//     resultsContainer.innerHTML = "";
//     summaryContainer.innerHTML = "";

//     getPredictions(file);
//   }

//   async function getPredictions(file) {
//     loader.style.display = "block";
//     const formData = new FormData();
//     formData.append("file", file);

//     try {
//       const response = await fetch("http://127.0.0.1:8000/predict/", {
//         method: "POST",
//         body: formData,
//       });

//       if (!response.ok)
//         throw new Error(`HTTP error! Status: ${response.status}`);

//       const data = await response.json();
//       displayResults(data.results);
//     } catch (error) {
//       console.error("Error getting predictions:", error);
//       resultsContainer.innerHTML = `<p style="color: red; text-align: center;">Failed to get predictions. Please ensure the backend server is running and try again.</p>`;
//     } finally {
//       loader.style.display = "none";
//     }
//   }

//   function displayResults(results) {
//     resultsContainer.innerHTML = "";
//     results.forEach((result) => {
//       const infoHtml = result.info
//         ? `
//         <div class="info-section">
//             <h4>Description</h4>
//             <p>${result.info.description}</p>
//             <h4>Common Cause</h4>
//             <p>${result.info.cause}</p>
//             <h4>Potential Treatment</h4>
//             <p>${result.info.treatment}</p>
//         </div>
//     `
//         : "<p>No detailed information available.</p>";

//       const cardHtml = `
//         <div class="result-card">
//             <h3>${result.model_name.replace(/_/g, " ").toUpperCase()}</h3>
//             <p class="prediction">${result.prediction}</p>
//             <p class="confidence-text" style="text-align:center; font-size: 0.8rem; opacity: 0.8;">Confidence: ${
//               result.confidence
//             }%</p>
//             <div class="confidence-bar-container">
//                 <div class="confidence-bar" style="width: ${
//                   result.confidence
//                 }%;"></div>
//             </div>
//             ${infoHtml}
//         </div>
//     `;
//       resultsContainer.innerHTML += cardHtml;
//     });

//     checkConsensus(results);
//   }

//   function checkConsensus(results) {
//     const predictions = results.map((r) => r.prediction);
//     const counts = {};
//     predictions.forEach((p) => {
//       counts[p] = (counts[p] || 0) + 1;
//     });

//     let consensusDiagnosis = null;
//     for (const diagnosis in counts) {
//       if (counts[diagnosis] >= 2) {
//         consensusDiagnosis = diagnosis;
//         break;
//       }
//     }

//     if (consensusDiagnosis) {
//       const button = document.createElement("button");
//       button.className = "summarise-btn";
//       button.textContent = "Summarise with AI";
//       button.onclick = () => getSummary(consensusDiagnosis);
//       summaryContainer.appendChild(button);
//     }
//   }

//   async function getSummary(diagnosis) {
//     summaryContainer.innerHTML = '<div class="loader"></div>';

//     try {
//       const response = await fetch("http://127.0.0.1:8000/summarise/", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ diagnosis: diagnosis }),
//       });
//       if (!response.ok) throw new Error("Failed to get summary.");

//       const data = await response.json();
//       const summaryHtml = `
//                 <div class="summary-box">
//                     <h3>AI Summary for ${diagnosis}</h3>
//                     <p>${marked.parse(data.summary)}</p>
//                 </div>
//             `;
//       summaryContainer.innerHTML = summaryHtml;
//     } catch (error) {
//       console.error("Error getting summary:", error);
//       summaryContainer.innerHTML = `<p style="color: red;">Failed to get AI summary.</p>`;
//     }
//   }
// }
document.addEventListener("DOMContentLoaded", () => {
  // --- MASTER INITIALIZATION FUNCTION ---
  // This runs once the entire page is ready.

  // 1. Initialize features common to all pages
  setupTheme();
  setupCursorEffect();
  initializeFloatingImages();

  // 2. Initialize features for specific pages
  if (document.querySelector(".slideshow-container")) {
    initializeSlideshow();
  }
  if (document.querySelector(".chat-container")) {
    initializeChatbot();
  }
  if (document.querySelector(".prediction-container")) {
    initializePredictionPage();
  }
  // Note: Add about.html slideshow initializer here if it uses this script
});


// --- FUNCTION DEFINITIONS ---

function setupTheme() {
    const themeToggle = document.getElementById("theme-toggle");
    const body = document.body;
    function applySavedTheme() {
        if (localStorage.getItem("theme") === "light") {
            body.classList.add("light-mode");
        }
    }
    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            body.classList.toggle("light-mode");
            localStorage.setItem("theme", body.classList.contains("light-mode") ? "light" : "dark");
        });
    }
    applySavedTheme();
}

function setupCursorEffect() {
    document.addEventListener("mousemove", (e) => {
        const cursorLight = document.querySelector(".cursor-light");
        if (cursorLight) {
            cursorLight.style.left = e.clientX + "px";
            cursorLight.style.top = e.clientY + "px";
        }
    });
}

function initializeFloatingImages() {
    let images = [];
    const imageElements = document.querySelectorAll(".floating-img");
    if (imageElements.length === 0) return;
    imageElements.forEach((img) => {
        images.push({
            element: img,
            x: Math.random() * (window.innerWidth - 150),
            y: Math.random() * (window.innerHeight - 150),
            dx: (Math.random() - 0.5) * 1.0,
            dy: (Math.random() - 0.5) * 1.0,
        });
    });
    function animate() {
        images.forEach((img) => {
            img.x += img.dx;
            img.y += img.dy;
            if (img.x <= 0 || img.x >= window.innerWidth - 150) img.dx *= -1;
            if (img.y <= 0 || img.y >= window.innerHeight - 150) img.dy *= -1;
            img.element.style.transform = `translate(${img.x}px, ${img.y}px)`;
        });
        requestAnimationFrame(animate);
    }
    animate();
}

// --- HOMEPAGE SLIDESHOW LOGIC ---
let slideIndex = 1;
let autoSlideTimeout;

function initializeSlideshow() {
    generateDots();
    showSlides(slideIndex);
}

function generateDots() {
    const slides = document.getElementsByClassName("slide-card");
    const dotsContainer = document.querySelector(".dots-container");
    if (!dotsContainer) return;
    dotsContainer.innerHTML = "";
    for (let i = 0; i < slides.length; i++) {
        const dot = document.createElement("span");
        dot.classList.add("dot");
        dot.setAttribute("onclick", `currentSlide(${i + 1})`);
        dotsContainer.appendChild(dot);
    }
}

function plusSlides(n) {
    clearTimeout(autoSlideTimeout);
    showSlides((slideIndex += n));
}

function currentSlide(n) {
    clearTimeout(autoSlideTimeout);
    showSlides((slideIndex = n));
}

function showSlides(n) {
    let i;
    let slides = document.getElementsByClassName("slide-card");
    let dots = document.getElementsByClassName("dot");
    if (slides.length === 0 || dots.length === 0) return;
    if (n > slides.length) { slideIndex = 1; }
    if (n < 1) { slideIndex = slides.length; }
    for (i = 0; i < slides.length; i++) {
        slides[i].classList.remove("active-slide");
        slides[i].style.display = "none";
    }
    for (i = 0; i < dots.length; i++) {
        dots[i].className = dots[i].className.replace(" active", "");
    }
    slides[slideIndex - 1].style.display = "flex";
    slides[slideIndex - 1].classList.add("active-slide");
    dots[slideIndex - 1].className += " active";
    autoSlideTimeout = setTimeout(() => plusSlides(1), 20000);
}


// --- CHATBOT LOGIC ---
function initializeChatbot() {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");
    const suggestionsContainer = document.getElementById("suggestion-cards");

    if (suggestionsContainer) {
        const suggestions = [
            "What is an Epidural Hemorrhage?",
            "Tell me about Subdural Hemorrhage.",
            "Explain Subarachnoid Hemorrhage.",
            "What are the signs of IPH?",
        ];
        suggestions.forEach((text) => {
            const card = document.createElement("div");
            card.className = "suggestion-card";
            card.textContent = text;
            card.onclick = () => {
                if(userInput) userInput.value = text;
                sendMessage();
            };
            suggestionsContainer.appendChild(card);
        });
    }

    function addMessageToBox(text, sender) {
        // ... (function remains the same)
    }

    async function sendMessage() {
        if (!userInput) return;
        const messageText = userInput.value.trim();
        if (!messageText) return;
        addMessageToBox(messageText, "user");
        userInput.value = "";
        if (suggestionsContainer) suggestionsContainer.style.display = "none";

        try {
            const response = await fetch("/summarise/", { // CORRECTED PATH
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ diagnosis: messageText }),
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            const botReply = data.summary || data.error || "Sorry, I couldn't get a response.";
            addMessageToBox(botReply, "bot");
        } catch (error) {
            console.error("Error fetching from API:", error);
            addMessageToBox("I'm having trouble connecting to my knowledge base. Please try again later.", "bot");
        }
    }
    
    if(sendButton) sendButton.addEventListener("click", sendMessage);
    if(userInput) userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendMessage();
    });

    addMessageToBox("Hello! I'm the TBiDx Assistant. How can I help you understand different types of Traumatic Brain Injuries today?", "bot");
}


// --- PREDICTION PAGE LOGIC ---
function initializePredictionPage() {
    const uploadBox = document.getElementById("upload-box");
    const imageUpload = document.getElementById("image-upload");
    const imagePreview = document.getElementById("image-preview");
    const resultsContainer = document.getElementById("results-container");
    const summaryContainer = document.getElementById("summary-container");
    const loader = document.getElementById("loader");

    if (imageUpload) imageUpload.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) handleFile(file);
    });

    if (uploadBox) {
        uploadBox.addEventListener("dragover", (e) => { e.preventDefault(); uploadBox.classList.add("dragover"); });
        uploadBox.addEventListener("dragleave", () => { uploadBox.classList.remove("dragover"); });
        uploadBox.addEventListener("drop", (e) => {
            e.preventDefault();
            uploadBox.classList.remove("dragover");
            const file = e.dataTransfer.files[0];
            if (file) handleFile(file);
        });
    }

    function handleFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            if(imagePreview) imagePreview.src = e.target.result;
            const previewContainer = document.getElementById("image-preview-container");
            if(previewContainer) previewContainer.style.display = "block";
            if(uploadBox) uploadBox.style.display = "none";
        };
        reader.readAsDataURL(file);
        if(resultsContainer) resultsContainer.innerHTML = "";
        if(summaryContainer) summaryContainer.innerHTML = "";
        getPredictions(file);
    }

    async function getPredictions(file) {
        if(loader) loader.style.display = "block";
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/predict/", { // CORRECTED PATH
                method: "POST",
                body: formData,
            });
            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
            const data = await response.json();
            displayResults(data.results);
        } catch (error) {
            console.error("Error getting predictions:", error);
            if(resultsContainer) resultsContainer.innerHTML = `<p style="color: red; text-align: center;">Failed to get predictions. Please try again.</p>`;
        } finally {
            if(loader) loader.style.display = "none";
        }
    }

    function displayResults(results) {
        // ... (function remains the same)
    }

    function checkConsensus(results) {
        // ... (function remains the same)
    }

    async function getSummary(diagnosis) {
        // ... (function remains the same, but with corrected fetch path)
        try {
            const response = await fetch("/summarise/", { // CORRECTED PATH
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ diagnosis: diagnosis }),
            });
            // ...
        } catch (error) {
            // ...
        }
    }
}
