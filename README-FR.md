# Hogwarts Legacy SpellCaster

Lancez des sorts dans Hogwarts Legacy à la voix. Dites *« Incendio »* et le
sort part, quel que soit le contenu de votre barre de sorts.

SpellCaster écoute votre micro, reconnaît l'incantation grâce à un modèle de
reconnaissance vocale entraîné sur le vocabulaire des sorts, puis lance le sort
dans le jeu par l'intermédiaire d'un mod [UE4SS](https://github.com/UE4SS-RE/RE-UE4SS).
Tout fonctionne en local sur votre machine.

*English version: [README.md](README.md).*

## Fonctionnement

```
 micro ─► application de bureau ─► mémoire partagée ─► mod UE4SS ─► jeu
          (VAD + Whisper ou Moonshine)                  (lance via WandTool)
```

L'application de bureau écoute et reconnaît. Elle transmet chaque nom de sort
reconnu au mod, qui le lance via le système de sorts du jeu lui-même. Comme le
mod pilote directement la logique du
jeu, il n'y a aucune touche ni aucune disposition de manette à configurer.

Deux anciens modes de sortie restent disponibles pour les installations sans le
mod : une manette Xbox virtuelle (ViGEm) qui appuie sur les boutons de votre
barre de sorts, et un plugin [UEVR](https://github.com/praydog/UEVR) pour jouer
en VR.

## Pour les joueurs

1. Téléchargez la dernière archive `HogwartsLegacy-SpellCaster-win-x64-<version>.zip`
   depuis la page [Releases](https://github.com/pierre-cheneau/HogwartsLegacy-SpellCaster/releases)
   et décompressez-la où vous voulez.
2. Lancez `HogwartsLegacy-SpellCaster.exe`. Au premier démarrage, l'application
   vous demande quel micro utiliser, si vous acceptez de contribuer des
   enregistrements pour améliorer les modèles, puis installe UE4SS et le mod
   SpellCaster dans le dossier du jeu. Les installations Steam sont détectées
   automatiquement ; sinon, renseignez `game_path` dans `config.ini` avec votre
   dossier `Hogwarts Legacy\Phoenix\Binaries\Win64`.
3. Lancez le jeu, chargez une sauvegarde, et parlez.

Prérequis : Windows 10 ou 11 en 64 bits et un micro. La reconnaissance tourne
sur le processeur par défaut ; une carte graphique compatible Vulkan accélère le
moteur Whisper. Le moteur se choisit avec `engine=whisper` ou
`engine=moonshine` dans `config.ini`.

Le dossier `spellbook` de l'archive montre chaque phrase reconnue avec sa
prononciation. En cas de problème, la sortie console de l'application et le
fichier `UE4SS_Logs\UE4SS.log` du dossier du jeu contiennent l'essentiel ;
apportez-les sur le Discord indiqué plus bas.

## Sorts

Tout ce que le mod sait lancer est listé dans
`ue4ss-mod/SpellCasterMod/dllmain.cpp`. En résumé :

| Groupe | Sorts |
|--------|-------|
| Contrôle | Accio, Levioso, Depulso, Descendo, Flipendo, Glacius, Arresto Momentum |
| Dégâts | Incendio, Confringo, Diffindo, Bombarda |
| Combat | Stupefy, Expelliarmus, Protego, Oppugno |
| Utilitaires | Lumos, Nox, Reparo, Revelio, Invisica (Désillusion), Wingardium Leviosa |
| Métamorphose | Transfigura Verto, Conjuration, Vanishment |
| Impardonnables | Avada Kedavra, Crucio, Imperio |
| Spéciaux | Smash (magie ancienne), Stealth Takedown, Confundo, Episkey |
| Actions | Finite, Appare Vestigium, Accio Balais / Broomstick, Accio Monture / Hippogriffe / Graphorn / Sombral |
| Menus | Apperta Codex, Meritas, Falcultates, Quaestiones, Mappa, Literae, Compendium, Incantatem, Configuratio |
| Packs de sorts communautaires | SpellsEnhanced, HRBSpellPack, HermitHollow et d'autres, s'ils sont installés |

## Organisation du dépôt

| Dossier | Contenu |
|---------|---------|
| `app/` | Application de bureau (Visual Studio 2022, C++20) : capture audio, VAD, moteurs Whisper et Moonshine, installateur UE4SS, client de collecte |
| `ue4ss-mod/` | Mod C++ UE4SS qui lance les sorts, plus un outil Python pour le piloter sans micro |
| `uevr-plugin/` | Plugin UEVR pour le mode manette en VR |
| `docs/` | Protocole inter-processus, mode de débogage audio, points connus à améliorer |

La documentation de développement est en anglais :
[BUILDING.md](BUILDING.md) pour compiler, [CONTRIBUTING.md](CONTRIBUTING.md)
pour ajouter des sorts ou contribuer, [docs/ipc-protocol.md](docs/ipc-protocol.md)
pour le dialogue entre l'application et le mod, [docs/known-issues.md](docs/known-issues.md)
pour ce qui reste à faire, [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
pour les licences des composants embarqués.

## Communauté et soutien

Créé par **Cornebidouil**. Questions, retours et enregistrements pour les
modèles sont les bienvenus :

- Discord : <https://discord.gg/zE4NRsTGdw>
- Portail d'entraînement (aidez à améliorer la reconnaissance) : <http://hogwartslegacyspellcaster.xyz>
- GitHub : <https://github.com/pierre-cheneau>

Si le projet vous plaît et que vous souhaitez soutenir son développement :
ETH `0x1F61fa7923d5E914A5Fdf36B584a1336fde20721`

## Licence

MIT, voir [LICENSE](LICENSE). Hogwarts Legacy est une marque de Warner Bros.
Entertainment Inc. ; ce projet est une création indépendante de fans.
