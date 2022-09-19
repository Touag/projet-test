const counterDisplay= document.querySelector('h3');
let counter = 0;
// logique du compteur


const chipolataBubble =()=>{
const bubble = document.createElement("span");
bubble.classList.add("bubble");
// on inject la classe de style a buuble
document.body.appendChild(bubble);
// on ajoute la span au body

// on crée un var pour la size random
const size = Math.random ()*200+100 +"px";
// on injecte la size randome height widht en PX
bubble.style.height = size ;
bubble.style.width= size ;

// on injecte le positionnement top et left en % on rajoute +50 pour que ça parte du bas

bubble.style.top = Math.random()*100+50+"%";
bubble.style.left = Math.random()*100+"%";
//on va injecter la left dynamic de manière random
// Maintenant qu'on a une bull opé  on va injecter plein de bubble
// on va incorporé du coup cette logique dans une fonction de type bubbleMaker
// on peut la call ChipolataBubble pour le fun
// on a probleme toute les bull parte a droite on a besoin d'ajouter du negatif dans la left
// fonction terner de type si math randome > 0.5 alors tu file 1 ou +1 selon
const plusMinus = Math.random () >0.5 ? 1:-1 ;
// on multiplie par -1 ou 1 en injectant la variable plusMinus
bubble.style.setProperty('--left',Math.random()*100 * plusMinus+"%");
// auto destruction de la bubble du body apres 8sec ( duré de l'annime CSs)
setTimeout(()=>{
    bubble.remove();
}, 8000);

// besoin => quand je click sur la buble ça disparait
// injection du counter

bubble.addEventListener('click',()=>{
    counter++;
    counterDisplay.textContent=counter;
    bubble.remove();
})

};
// la fonction ne se joue pas toute seul maintenant car en mode fonction

setInterval(chipolataBubble,300);


