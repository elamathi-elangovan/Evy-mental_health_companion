document.getElementById("send-btn").addEventListener("click", sendMessage);
document.getElementById("user-input").addEventListener("keypress", function(e) {
  if (e.key === "Enter") sendMessage();
});

function sendMessage() {
  let inputField = document.getElementById("user-input");
  let userMessage = inputField.value.trim();
  if (!userMessage) return;

  // Add user message
  appendMessage(userMessage, "user-message");
  inputField.value = "";

  // Send to backend
  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: userMessage })
  })
  .then(res => res.json())
  .then(data => {
    let botReply = data.chatbot_reply || "I'm here to listen 💙";
    appendMessage(botReply, "bot-message");
  })
  .catch(err => {
    appendMessage("⚠️ Something went wrong. Please try again.", "bot-message");
    console.error(err);
  });
}

function appendMessage(text, className) {
  let chatBox = document.getElementById("chat-box");
  let messageDiv = document.createElement("div");
  messageDiv.classList.add("message", className);
  messageDiv.textContent = text;
  chatBox.appendChild(messageDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}
