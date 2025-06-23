
const dropZone = document.getElementById('dropZone');
const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');
const form = document.getElementById('uploadForm');
const loader = document.getElementById('loader');

// Clicking the drop zone triggers file input
dropZone.addEventListener('click', () => imageInput.click());

// Preview image before submission
imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = () => {
      previewImage.src = reader.result;
      previewImage.style.display = 'block';
    };
    reader.readAsDataURL(file);
  }
});

// Show loader on submit
form.addEventListener('submit', () => {
  loader.style.display = 'block';
});
