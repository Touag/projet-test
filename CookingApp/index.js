// https://www.themealdb.com/api/json/v1/1/search.php?s=tomato
// sort que les recette de tomate , il faudra rendre la fin dynamique une fois qu'on a la logique
const result = document.getElementById("result");
const form = document.querySelector("form");
const input = document.querySelector("input");
let meals = [];

async function fetchMeal(search) {
  await fetch(
    /*"https://www.themealdb.com/api/json/v1/1/search.php?s="+search*/ `https://www.themealdb.com/api/json/v1/1/search.php?s=${search}`
  )
    .then((res) => res.json())
    .then((data) => (meals = data.meals));
  console.log(meals);
}

function mealsDisplay() {
  if (meals === null) {
    result.innerHTML = "<h2>Aucun résultat pour votre recherche :( !</h2>";
  }
  meals.length = 12;
  result.innerHTML = meals
    .map((meal) => {
      let ingredients = [];
      for (let i = 1; i < 21; i++) {
        if (meal[`strIngredient${i}`]) {
          let ingredient = meal[`strIngredient${i}`];
          let measure = meal[`strMeasure${i}`];

          ingredients.push(`<li>${ingredient} -${measure}</li>`);
        }
      }
      console.log(ingredients);

      return `
        <li class="card">
        <h2>${meal.strMeal}</h2>
        <p>${meal.strArea}</p>
        <img src=${meal.strMealThumb} alt ="photo of ${meal.strMeal}">
        <ul>${ingredients.join("")}</ul>
        </li>
        
        `;
    })
    .join(" ");
}
// faut qu'on recupere ce qui saisie dans linput avec un e.target.value pour l'injecter dans la recherche sur le lien api
input.addEventListener("input", (e) => {
  fetchMeal(e.target.value);
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  mealsDisplay();
});
