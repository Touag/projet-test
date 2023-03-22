from pytube import YouTube


# verrifier que l'url commence par youtube
# si ce n'est pas le cas print erreur
def get_url_from_user():
    base_youtube="https://www.youtube.com/" 

    while True:
        url=input("Copier-coller le liens http de la video youtube que vous voulez Download: ")
        #autre méthode url[:len(base_youtube)]==base_youtube:
        if url.lower().startswith(base_youtube):
                break
        print("ERREUR:Il faut obligatoirement mettre un lien youtube qui commence par https://www.youtube.com/....")

    return url
    







def ondownload_progress(stream, chunk,bytes_remaining):
        bytes_dl=stream.filesize-bytes_remaining
        percent=(bytes_dl*100/stream.filesize)
        print("progression du téléchargement : ",int(percent),"%")
        # on pose le int sur le percent pour suprimer les chiffre a virgule lors de l'affichage


    # Demander le choix de la resolution

def get_choice_from_user(streams):
    print("")
    print(" Vous avez choisi le titre : "+youtube_video.title)
    print("")
    print("Différente qualité de stream disponible:")
    j=1
    for stream in streams:
        print(j,")",stream.resolution)
        j+=1

    while True:
        choice=input(f"Quelle est votre choix ? (merci d'entrée un chiffre entre 1 et {len(streams)}) : ")
        if choice=="":
            print("Erreur:vous devez entrer  un nombre")
        else:
            try:
                choice_int=int(choice)
            except:
                print("Erreur: Veuillez entrez un nombre correspondant à la résolution")
            else:
                if not 1<=choice_int<=len(streams):
                    print(" vous devez entrez un nombre compris dans les chiffres proposer soit entre 1 et ",len(streams))
                else:
                    break
    print("Vous avez selectionné la résolution suivante",streams[choice_int-1].resolution)
    print("")
    print("Le téléchargement va se lancer")
    itag_select=streams[choice_int-1].itag
    return itag_select


"""---------Programme principale---------"""

"""url="https://www.youtube.com/watch?v=ws3WGmINlIg"""
url=get_url_from_user()

youtube_video=YouTube(url)

youtube_video.register_on_progress_callback(ondownload_progress)

streams=youtube_video.streams.filter(progressive=True)

itag_select2=get_choice_from_user(streams)

stream = youtube_video.streams.get_by_itag(itag_select2)
stream.download()
print("Téléchargement fini! vous pouvez kiffer")
