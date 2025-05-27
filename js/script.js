const form = document.getElementById('loginForm');
const erroMsg = document.getElementById('erroMsg');

form.addEventListener('submit', function(e) {
  e.preventDefault();
  const email = document.getElementById('email').value.trim();
  const senha = document.getElementById('senha').value.trim();

  if (!email || !senha || senha.length < 6) {
    erroMsg.style.display = 'block';
  } else {
    erroMsg.style.display = 'none';
    window.location.href = "mainpage.html";
  }
});