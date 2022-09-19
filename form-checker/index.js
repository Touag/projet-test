const inputs = document.querySelectorAll('input[type="text"],input[type="password"]');
const form = document.querySelector('form');
const progressBar =document.getElementById('progress-bar');
// on va crée des variable pour stocker les donnée des inputs du formulaire
let pseudo, email, password ,confirmPass

const errorDisplay=(tag, message,valid)=>{ 
  const container = document.querySelector('.'+tag+'-container');// concataine pour avoir un paramètre dynamique dans la fonction
  const span = document.querySelector('.'+tag+'-container>span');//ça évite la répétition de code comme en bas
  if (!valid){
    container.classList.add ('error');
    span.textContent= message;

  }
else{
  container.classList.remove('error');
  span.textContent= '';

}
}
const pseudoChecker = (value)=>{
  if(value.length>0 && value.length<3|| value.length>20){
    errorDisplay("pseudo","Le pseudo doit faire entre 3 et 20 caractères");
    pseudo=null;
  }else if (!value.match(/^[a-zA-Z0-9_.-]*$/)){
    errorDisplay("pseudo","Le pseudo ne doit pas contenir de caractère spéciaux %$!?")
    pseudo=null;
  } else {
    errorDisplay('pseudo','',true)
    pseudo=value;
  }

 /* const pseudoContainer = document.querySelector('.pseudo-container');
  const errorDisplay = document.querySelector('.pseudo-container>span');

  if(value.length>0 && value.length<3|| value.length>20){
    pseudoContainer.classList.add ("error");// class codé en css
    errorDisplay.textContent="Le pseudo doit faire entre 3 et 20 caractère!";
  }else if (!value.match(/^[a-zA-Z0-9_.-]*$/)){
    pseudoContainer.classList.add ("error");
    errorDisplay.textContent='Le pseudo ne doit pas contenir de caractère spéciaux %µ$!?';
  } else {
    pseudoContainer.classList.remove('error');
    errorDisplay.textContent ='';

    pour evité d'avoir un code répétitif on peu use une fonction d'afichage qu'on intro dans la fonction 
    */
  }
;

const emailChecker =(value)=>{
   if(!value.match(/^[\w_-]+@[\w-]+\.[a-z]{2,4}$/i)){
      errorDisplay('email',"Le mail n'est pas valide!",);
      email=null;   }
   else{
    errorDisplay('email','',true)
    email=value;// on incremente la variable pour stocker la donnée;)
   }
};

const passwordChecker =(value) =>{
  progressBar.classList =""; // on ecrit ça pour évité un empilement de classe et que ça revienne a zero
   if (!value.match(/^(?=.*?[A-Z])(?=(.*[a-z]){1,})(?=(.*[\d]){1,})(?=(.*[\W]){1,})(?!.*\s).{8,}$/)){
    errorDisplay('password','Minimum de 8 caractère,une majuscule, un chiffre et un caractère spécial')
    password=null;
  progressBar.classList.add('progressRed');
  } else if (value.length <12){
    progressBar.classList.add('progressBlue');
    errorDisplay('password','',true);
    password=value;
  }
  
  else {
    errorDisplay('password','',true);
    progressBar.classList.add('progressGreen');
    password=value;
  }
  if(confirmPass)confirmChecker(confirmPass);
   
};

const confirmChecker =(value) =>{

   if ( value !== password){
    errorDisplay('confirm','Mot de passe saisie différent');
    confirmPass=false;// dans la logique on sen fou de stoker une deuxieme fois le MDp 
   }else{
    errorDisplay('confirm','',true);
    confirmPass=true;
   }
};

inputs.forEach((input) => {
  input.addEventListener("input", (e) => {
    switch (e.target.id) {
      case "pseudo":
        pseudoChecker(e.target.value);
        break;
      case "email":
        emailChecker(e.target.value);
        break;
      case "password":
        passwordChecker(e.target.value);
        break;
      case "confirm":
        confirmChecker(e.target.value);
        break;
      default:
        nul ;
    }
  });
});

// envoi du formulaire
form.addEventListener('submit',(e)=>{
  e.preventDefault();// evite le chargement de page quand on soumet le formulaire

  if ( pseudo && email&& password&&confirmPass){
    const data= {
      pseudo :pseudo,
      email : email,
      password:password,
    } 

    inputs.forEach((input) => (input.value=''));
    progressBar.classList='';
    password=null;
    email=null;
    pseudo=null;
    confirmPass=null;
    alert('Inscription validée !')
    console.log(data);
      
    
  }else{
    alert('Veuillez remplire correctement les différents champs!')
  }
})

  