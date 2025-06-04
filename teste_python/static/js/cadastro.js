const cadastroForm = document.getElementById('cadastroForm');
const cadastroErroMsg = document.getElementById('cadastroErroMsg');

cadastroForm.addEventListener('submit', function(e) {
  e.preventDefault();
  const nome = document.getElementById('cadastroNome').value.trim();
  const email = document.getElementById('cadastroEmail').value.trim();
  const senha = document.getElementById('cadastroSenha').value.trim();
  const confirmar = document.getElementById('cadastroConfirmarSenha').value.trim();

  if (!nome || !email || senha.length < 6 || senha !== confirmar) {
    cadastroErroMsg.textContent = 'Verifique os campos e certifique-se que as senhas coincidem.';
    cadastroErroMsg.style.display = 'block';
  } else {
    cadastroErroMsg.style.display = 'none';
    window.location.href = "mainpage.html";
  }
});