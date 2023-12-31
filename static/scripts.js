document.addEventListener('DOMContentLoaded', function () {
    
    const resultContainer = document.querySelector('.result-container');
    const resultCard = document.createElement('div');
    resultCard.classList.add('result-card');

    const predictionElement = document.createElement('p');
    predictionElement.innerHTML = `<strong>Prediction:</strong> ${result.Prediction}`;
    const freshnessElement = document.createElement('p');
    freshnessElement.innerHTML = `<strong>Freshness:</strong> ${result.Freshness}`;
    const typeElement = document.createElement('p');
    typeElement.innerHTML = `<strong>Type:</strong> ${result.Type}`;

    resultCard.appendChild(predictionElement);
    resultCard.appendChild(freshnessElement);
    resultCard.appendChild(typeElement);

    resultContainer.appendChild(resultCard);
});
