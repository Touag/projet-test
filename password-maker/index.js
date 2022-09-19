// stocker les donnée possible dans un tableau
const dataLowercase ="azertyuiopqsdfghjklmwxcvbn";
// const dataUppercase= dataLowercase.toUpperCase();
const dataUppercase ="AZERTYUIOPQSDFGHJKLMWXCVBN";
const dataNumbers="0123456789";
const dataSymbols="&~#!§%*+-/$%?";
const rangeValue = document.getElementById('password-length');
const passwordOutput = document.getElementById('password-output');



function generatePassword(){
    let data =[];
    let password="";
    if (lowercase.checked)data.push(...dataLowercase )
    if (uppercase.checked)data.push(...dataUppercase )
    if (number.checked)data.push(...dataNumbers)
    if (symbols.checked)data.push(...dataSymbols )
    if (data.length=== 0){
        alert('Veuillez sélectionner des critères');
        return;
    }

     for( i=0 ; i< rangeValue.value ; i++){
        password += data[Math.floor(Math.random()*data.length)];

        // attention use += sinon ça s'efface avec =
        console.log(password);

     }
     // sur input on peut pas innerhtml ou un textcontent
     passwordOutput.value= password;

     passwordOutput.select();
     document.execCommand("copy");

     generateButton.textContent ="Copié!";
     
     setTimeout(()=>{
        generateButton.textContent='Générer un nouveau mot de passe'
     },2000);

}

generateButton.addEventListener("click",generatePassword);