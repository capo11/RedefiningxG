import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
from mplsoccer import VerticalPitch
import joblib
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import xgboost
import shap
from streamlit_option_menu import option_menu

shotsMultiplier = 500
lastRound = 18

pd.options.mode.chained_assignment = None
# st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("""
    <style>
    .card {
        background-color:rgb(0, 8, 255); /* Colore sfondo card */
        border: 2px solidrgb(239, 239, 239); /* Colore bordo */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        max-width: 300px;
        margin: auto;
        font-family: Arial, sans-serif;
        text-align: center;
        height: 90%;
    }
    .card img {
        border-radius: 10px;
        width: 100%;
        height: auto;
        margin-bottom: 15px;
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 15px;
        color: rgb(255, 255, 255);
    }
    .card-row {
        display: flex;
        justify-content: space-between;
        font-size: 1rem;
        margin: 10px 0;
        color: rgb(255, 255, 255);
    }
    .card-difference {
        font-size: 1.2rem;
        font-weight: bold;
        color:rgb(255, 255, 255); /* Rosso */
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

def cleanDataset(df, elo=False, minute=True):
  df = df.dropna(subset=['xg'])
  df = df.loc[df['position'] != 'G']
  # One-Hot Encoding
  x = pd.get_dummies(df, columns=['position'], prefix='position', drop_first=False)
  x = pd.get_dummies(x, columns=['situation'], prefix='situation', drop_first=False)
  x = pd.get_dummies(x, columns=['bodyPart'], prefix='bodyPart', drop_first=False)



  # Normalizations
  if minute == True:
    x['minute'] = x['minute']/90
  x['rating'] = x['rating']/100
  if elo == True:
    x['eloTeam'] = x['eloTeam']/2168
  x['keeperRating'] = (x['keeperRating'].astype(int))/100
  if elo == True:
    x['eloOpponent'] = x['eloOpponent']/2168
  x['distance'] = x['distance']/120
  x['angle'] = x['angle']/90

  # Conversions
  x['position_D'] = x['position_D'].astype(int) # Defender
  x['position_F'] = x['position_F'].astype(int) # Forward
  x['position_M'] = x['position_M'].astype(int) # Midfielder

  x['situation_assisted'] = x['situation_assisted'].astype(int)
  x['situation_corner'] = x['situation_corner'].astype(int)
  x['situation_fast-break'] = x['situation_fast-break'].astype(int)
  x['situation_free-kick'] = x['situation_free-kick'].astype(int)
  x['situation_penalty'] = x['situation_penalty'].astype(int)
  x['situation_regular'] = x['situation_regular'].astype(int)
  x['situation_set-piece'] = x['situation_set-piece'].astype(int)
  x['situation_throw-in-set-piece'] = x['situation_throw-in-set-piece'].astype(int)

  x['bodyPart_head'] = x['bodyPart_head'].astype(int)
  x['bodyPart_weak-foot'] = x['bodyPart_weak-foot'].astype(int)
  x['bodyPart_other'] = x['bodyPart_other'].astype(int)
  x['bodyPart_strong-foot'] = x['bodyPart_strong-foot'].astype(int)

  y = df['goal']
  x['isHome'] = x['isHome'].astype(int)
  x = x.drop(columns=['goal', 'player', 'team', 'keeper', 'opponent', 'x', 'y', 'xg', 'homeTeam', 'awayTeam', 'index', 'round'])
  return x, y

def getXTrain(df, elo=False, minute=True, over='none', k=15, sampling_strategy='none', test_size=0.2):
    df = df.dropna(subset=['xg'])
    x,y = cleanDataset(df, elo=elo, minute=minute)
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=test_size, random_state=42)
    if(over == "random"):
        oversample = RandomOverSampler(sampling_strategy=0.25, random_state=42)
        x,y = oversample.fit_resample(x,y)
    elif(over == "smote"):
        if(k!=0):
            smt = SMOTE(k_neighbors=k, random_state=42)
        else:
            if(sampling_strategy=='none'):
                smt = SMOTE(random_state=42)
            else:
                smt = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        x,y = smt.fit_resample(x,y)
    elif(over == "adasyn"):
        ada = ADASYN(random_state=42)
        x,y = ada.fit_resample(x,y)
    
    
    return X_train, x

def predictLocalGame(game, model, elo=False, minute=True):
  allShots = pd.read_csv('seriea2425_id.csv')
  allShots = allShots.drop(columns=['playerID', 'keeperID'])
  shotmap = allShots.loc[(allShots['homeTeam'] == game['home_team']) & (allShots['awayTeam'] == game['away_team'])]
  shotmap = shotmap.reset_index()
  shotmap = shotmap.drop(columns=['level_0'])
  if minute==False:
    shotmap = shotmap.drop(columns=['minute'])
  shotmap = shotmap.drop(columns=['index', 'round', 'homeTeam', 'awayTeam'])
  shotmap = shotmap.dropna(subset=['xg'])
  shotmap = shotmap.loc[shotmap['position'] != 'G']
  if elo==False:
    shotmap = shotmap.drop(columns=['eloTeam', 'eloOpponent'])

  homeShots = shotmap.loc[shotmap['isHome'] == True]
  homeShots = homeShots.reset_index()
  homeShots = homeShots.drop(columns=['index'])
  awayShots = shotmap.loc[shotmap['isHome'] == False]
  awayShots = awayShots.reset_index()
  awayShots = awayShots.drop(columns=['index'])

  df = pd.read_csv('seriea_joined_new.csv')
  df = df.drop(columns=['Unnamed: 0'])
  if minute==False:
    df = df.drop(columns=['minute'])
  if elo==False:
    df = df.drop(columns=['eloTeam', 'eloOpponent'])
  df_homeShots = pd.concat([df, homeShots]).reset_index()
  df_homeShots = df_homeShots.drop(columns=['level_0'])
  df_x, df_y = cleanDataset(df_homeShots, elo=elo, minute=minute)

  homeShots_clean = df_x.loc[len(df):]
#   homeShots_clean = homeShots_clean.drop(columns=['index'])
  homeShots_clean = homeShots_clean.reset_index()
  homeShots_clean = homeShots_clean.drop(columns=['index'])
  homeXgPred = model.predict_proba(homeShots_clean)[:, 1]

  homePred = model.predict(homeShots_clean)
  homeShots['goalPred'] = homePred
  homeShots['xgPred'] = homeXgPred
  for i in homeShots.index:
    if (homeShots.loc[i]['situation'] == 'penalty'):
      homeShots.at[i, 'xgPred'] = 0.75
    if (homeShots.loc[i]['xgPred'] == 0):
      homeShots.at[i, 'xgPred'] = 0.01
  homeShots['diff'] = homeShots['xgPred']-homeShots['xg']

  df_awayShots = pd.concat([df, awayShots]).reset_index()
  df_awayShots = df_awayShots.drop(columns=['level_0'])
  df_x, df_y = cleanDataset(df_awayShots, elo=elo, minute=minute)
  awayShots_clean = df_x.loc[len(df):]
#   awayShots_clean = awayShots_clean.drop(columns=['index'])
  awayShots_clean = awayShots_clean.reset_index()
  awayShots_clean = awayShots_clean.drop(columns=['index'])
  awayXgPred = model.predict_proba(awayShots_clean)[:, 1]
  awayPred = model.predict(awayShots_clean)
  awayShots['goalPred'] = awayPred
  awayShots['xgPred'] = awayXgPred
  for i in awayShots.index:
    if (awayShots.loc[i]['situation'] == 'penalty'):
      awayShots.at[i, 'xgPred'] = 0.75
    if (awayShots.loc[i]['xgPred'] == 0):
      awayShots.at[i, 'xgPred'] = 0.01
  awayShots['diff'] = awayShots['xgPred']-awayShots['xg']





  stats = {
      "shotmap": shotmap,
      "homeShots": homeShots,
      "awayShots": awayShots,
      "homeShots_clean": homeShots_clean,
      "awayShots_clean": awayShots_clean,
      "homeXgPred": homeXgPred,
      "homePred": homePred,
      "awayXgPred": awayXgPred,
      "awayPred": awayPred
  }
  return stats

def plotShots(teamShots):
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#22312b', half=True)
    fig,axs = pitch.draw(figsize=(8,4), ncols=2)
    fig.set_facecolor('#22312b')
    axs[0].patch.set_facecolor('#22312b')
    axs[0].set_title("xG Sofascore", color="white")
    axs[1].patch.set_facecolor('#22312b')
    axs[1].set_title("xG Modello", color="white")

    legend1 = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Piede Forte'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Piede Debole'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Testa'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Altro')
    ]
    
    legend2 = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='xG Modello > xG Sofascore'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='xG Modello = xG Sofascore'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='xG Modello < xG Sofascore')
    ]

    axs[0].legend(handles=legend1, loc='lower center', title='Legenda')
    axs[1].legend(handles=legend2, loc='lower center', title='Legenda')
    descriptions = []
    for i in teamShots.index:
        if(teamShots.loc[i]['goal'] == 0):
            shotOutcome = 'No Goal'
        else:
            shotOutcome = 'Goal'
        description = str(i+1) + ' - ' + teamShots.loc[i]['player'] + ' - ' + shotOutcome + ' (' + str(teamShots.loc[i]['xg']) + ' xG)'
        # print(shotDescription)
        descriptions.append(description)
        x = 120-teamShots.loc[i]['x']
        y = teamShots.loc[i]['y']
        if(teamShots.loc[i]['bodyPart'] == 'strong-foot'):
            color='green'
        elif(teamShots.loc[i]['bodyPart'] == 'weak-foot'):
            color='red'
        elif(teamShots.loc[i]['bodyPart'] == 'head'):
            color='yellow'
        elif(teamShots.loc[i]['bodyPart'] == 'other'):
            color='blue'
        pitch.scatter(
            x=x, 
            y=y,
            ax=axs[0],
            s = shotsMultiplier*teamShots.loc[i]['xg'],
            c=color,
            edgecolors='white')
        if(teamShots.loc[i]['xg']<teamShots.loc[i]['xgPred']):
            color='green'
        elif(teamShots.loc[i]['xg']>teamShots.loc[i]['xgPred']):
            color='red'
        else:
            color='yellow'
        pitch.scatter(
            x=x, 
            y=y,
            ax=axs[1],
            s = shotsMultiplier*teamShots.loc[i]['xgPred'],
            c=color,
            edgecolors='white')
    st.pyplot(fig)
    return descriptions

def plotShap(shapValues, elo):
    features = []
    shap_values = []
    values = []
    for (i, feature) in enumerate(shapValues.data.index):
        # print(i,feature)
        features.append(feature)
        shap_values.append(round(shapValues.values[i], 2))
    if elo == True:
        for (i, value) in enumerate(shapValues.data):
            # print(i, value)
            if(i==0):
                value = int(value*90)   #minuto
            elif (i==1):
                value = int(value)  #differenza goal
            elif(i==2 or i==4):
                value = int(value*100)  #ratings
            elif(i==3 or i==5):
                value = int(value*2168) #elos
            elif(i==7):
                value = value*120   #distanza
            elif(i==8):
                value = value*90    #angolo
            else:
                value = int(value)
            values.append(value)
    else:
       for (i, value) in enumerate(shapValues.data):
            # print(i, value)
            if(i==0):
                value = int(value*90)   #minuto
            elif (i==1):
                value = int(value)  #differenza goal
            elif(i==2 or i==3):
                value = int(value*100)  #ratings
            elif(i==5):
                value = int(value*120)   #distanza
            elif(i==6):
                value = int(value*90)    #angolo
            else:
                value = int(value)
            values.append(value)
    # print(features)
    # print(values)
    if(elo == True):
        features = ['Minuto','Differenza Goal', 'Rating Tiratore', 'Elo Squadra', 'Rating Portiere', 'Elo Avversario', 'Gioca in Casa', 'Distanza', 'Angolo', 'Posizione - Difensore', 'Posizione - Attaccante', 'Posizione - Centrocampista', 'Situazione - Servito', 'Situazione - Corner', 'Situazione - Contropiede', 'Situazione - Punizione', 'Situazione - Rigore', 'Situazione - Regolare', 'Situazione - Calcio Piazzato', 'Situazione - Rimessa Laterale', 'Corpo - Testa', 'Corpo - Altro', 'Corpo - Piede Forte', 'Corpo - Piede Debole']
    else:
        features = ['Minuto','Differenza Goal', 'Rating Tiratore', 'Rating Portiere', 'Gioca in Casa', 'Distanza', 'Angolo', 'Posizione - Difensore', 'Posizione - Attaccante', 'Posizione - Centrocampista', 'Situazione - Servito', 'Situazione - Corner', 'Situazione - Contropiede', 'Situazione - Punizione', 'Situazione - Rigore', 'Situazione - Regolare', 'Situazione - Calcio Piazzato', 'Situazione - Rimessa Laterale', 'Corpo - Testa', 'Corpo - Altro', 'Corpo - Piede Forte', 'Corpo - Piede Debole']
    features_values = []
    if(elo==True):
        for i in range(0, len(features)):
            if(i<=8 and i!=6):
                features_values.append(str(features[i]) + ': ' + str(round(values[i], 2)))
            else:
                if(values[i] == 1):
                    features_values.append(str(features[i]) + ': Si')
                else:
                    features_values.append(str(features[i]) + ': No')
    else:
        for i in range(0, len(features)):
            if(i<=6 and i!=4):
                features_values.append(str(features[i]) + ': ' + str(round(values[i], 2)))
            else:
                if(values[i] == 1):
                    features_values.append(str(features[i]) + ': Si')
                else:
                    features_values.append(str(features[i]) + ': No')
    # print(features_values)

    shap_values = np.array(shap_values)
    sorted_indices = np.argsort(np.abs(shap_values))[10:]
    sorted_shap_values = shap_values[sorted_indices]
    sorted_feature_names = [features_values[i] for i in sorted_indices]
    fig = plt.figure(figsize=(8, 6))
    bars = plt.barh(sorted_feature_names, sorted_shap_values, color=["green" if v > 0 else "red" for v in sorted_shap_values])
    plt.xlabel("Shapley Value", color="white")
    plt.ylabel("Feature", color="white")
    plt.title("Quali fattori influenzano il tiro?", color="white")
    plt.axvline(0, color="white", linewidth=0.8, linestyle="--")  # Linea verticale per il riferimento a zero
    # plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tick_params(axis='both', colors='white')
    for pos in ['right', 'top', 'bottom', 'left']: 
        plt.gca().spines[pos].set_visible(False) 


    # Mostrare il grafico
    # plt.show()
    st.pyplot(fig, transparent=True)

    
    # plotData = pd.DataFrame({
    #     "Feature": sorted_feature_names,
    #     "Shapley Value": sorted_shap_values
    # })
    # fig = px.bar(
    #     plotData,
    #     x="Shapley Value",
    #     y="Feature",
    #     orientation='h',
    #     color="Shapley Value",
    #     color_continuous_scale=["red", "green"]
    # )
    # fig.update_layout(
    #     yaxis=dict(
    #         title="Feature",  # Se vuoi mantenere il titolo, altrimenti rimuovilo
    #         color="white",  # Colore del titolo dell'asse
    #         showticklabels=False  # Nasconde le etichette sull'asse Y
    #     ),
    #     plot_bgcolor='rgba(0,0,0,0)',  # Sfondo del grafico trasparente
    #     paper_bgcolor='rgba(0,0,0,0)',  # Sfondo del foglio trasparente
    #     font=dict(color="white"),  # Colore del testo
    #     xaxis=dict(title="Shapley Value", color="white"),
    #     title=dict(
    #         text="Quali fattori influenzano il tiro?",
    #         x=0.5,  # Centra il titolo orizzontalmente
    #         xanchor='center',  # Assicura l'ancoraggio al centro
    #         font=dict(size=20, color="white")  # Dimensione e colore del titolo
    #     )
    # )
    # st.plotly_chart(fig)

def showShots():
    df = pd.read_csv('seriea_joined_new.csv')
    df = df.drop(columns=['Unnamed: 0'])
    useElo = st.checkbox("Utilizza anche gli Elo Rating delle squadre")
    if useElo == True:
        elo = True
        modelName = 'base_XGB_full'
    else:
        elo = False
        modelName = 'base_XGB_minute'
        df = df.drop(columns=['eloTeam', 'eloOpponent'])
    model = joblib.load('models/' + modelName + '.sav')

    X_train, X = getXTrain(df, elo=elo, minute=True)
    explainer = shap.Explainer(model, X_train)





    shotsDF = pd.read_excel('allShots/allShots_' + modelName + '.xlsx')
    shotsDF = shotsDF.drop(columns='Unnamed: 0')
    statsDF = pd.read_excel('leagueStats/leagueStats_' + modelName + '.xlsx')
    statsDF = statsDF.drop(columns='Unnamed: 0')


    schedule = pd.read_csv('serieaSchedule.csv')

    
    teams = np.unique(schedule['home_team'])
    scheduleTeam = st.selectbox("Seleziona una Squadra", teams, index=None)
    if scheduleTeam:
        schedule = schedule.drop(columns='Unnamed: 0')
        scheduleDone = schedule.loc[schedule['week']<=lastRound]
        scheduleDone = scheduleDone.loc[(scheduleDone['home_team'] == scheduleTeam) | (scheduleDone['away_team'] == scheduleTeam)]
        descriptions = []
        for i in scheduleDone.index:
            description = str(scheduleDone.loc[i]['week']) + '° Giornata: ' + scheduleDone.loc[i]['home_team'] + ' - ' + scheduleDone.loc[i]['away_team'] + ' ' + str(int(scheduleDone.loc[i]['home_score'])) + ' - ' + str(int(scheduleDone.loc[i]['away_score']))
            descriptions.append(description)
        scheduleDone['description'] = descriptions
        gameDescription = st.selectbox('Seleziona una Partita', scheduleDone['description'], index=None)

        if gameDescription:
            gameIndex = scheduleDone.loc[scheduleDone['description'] == gameDescription].index[0]
            st.error("xG Sofascore: " + str(statsDF.loc[gameIndex]['homeXg']) + ' - ' + str(statsDF.loc[gameIndex]['awayXg']))
            st.info("xG Previsti dal Modello: " + str(statsDF.loc[gameIndex]['homeXgPred']) + ' - ' + str(statsDF.loc[gameIndex]['awayXgPred']))
            stats = predictLocalGame(scheduleDone.loc[gameIndex], model, elo=elo, minute=True)
            gameShots = shotsDF.loc[shotsDF['gameIndex'] == gameIndex]
            
            
            home_team = scheduleDone.loc[gameIndex]['home_team']
            away_team = scheduleDone.loc[gameIndex]['away_team']

            selectedTeam = st.selectbox('Seleziona una Squadra', [home_team, away_team], index=None)

            if selectedTeam:
                teamShots = gameShots.loc[gameShots['team'] == selectedTeam].reset_index(drop=True)
            
                teamShots['description'] = plotShots(teamShots)
                shotDescription = st.selectbox('Seleziona un Tiro', teamShots['description'], index=None)
                if shotDescription:
                    shotIndex = teamShots.loc[teamShots['description'] == shotDescription].index[0]
                    
                    st.error("xG Sofascore: " + str(teamShots.loc[shotIndex]['xg']))
                    st.info("xG Previsto dal Modello: " + str(teamShots.loc[shotIndex]['xgPred']))
                    
                    if(selectedTeam == home_team):
                        shot = stats['homeShots_clean'].loc[shotIndex]
                    elif(selectedTeam == away_team):
                        print(stats['awayShots_clean'])
                        shot = stats['awayShots_clean'].loc[shotIndex]
                    shapValues = explainer(shot)
                    plotShap(shapValues, elo)


def showPlayers():
    useElo = st.checkbox("Utilizza anche gli Elo Rating delle squadre")
    if useElo == True:
        elo = True
        modelName = 'base_XGB_full'
    else:
        elo = False
        modelName = 'base_XGB_minute'

    shotsDF = pd.read_excel('allShots/allShots_' + modelName + '.xlsx')
    shotsDF = shotsDF.drop(columns=['Unnamed: 0'])
    
    photoStrikers(shotsDF)
    photoKeepers(shotsDF)

    

def photoStrikers(shotsDF):
    shotPlayers = np.unique(shotsDF['player'])
    players = []
    playerIDs = []
    xgSums = []
    xgPredSums = []
    goalSums = []
    for player in shotPlayers:
        playerShots = shotsDF.loc[shotsDF['player'] == player].reset_index()
        # print(playerShots)
        playerID = playerShots.loc[0]['playerID']
        xgSum = np.sum(playerShots['xg'])
        xgPredSum = np.sum(playerShots['xgPred'])
        goalSum = np.sum(playerShots['goal'])
        players.append(player)
        playerIDs.append(playerID)
        xgSums.append(xgSum)
        xgPredSums.append(xgPredSum)
        goalSums.append(goalSum)
    playersDF = pd.DataFrame()
    playersDF['Giocatore'] = players
    playersDF['ID Giocatore'] = playerIDs
    playersDF['xG Sofascore'] = xgSums
    playersDF['xG Modello'] = xgPredSums
    playersDF['Goal'] = goalSums
    playersDF['Differenza (Sofascore)'] = playersDF['Goal'] - playersDF['xG Sofascore']
    playersDF['Differenza (Modello)'] = playersDF['Goal'] - playersDF['xG Modello']

    st.header("Giocatori di Movimento")
    col1, col2 = st.columns(2)
    sofaOPdf = playersDF.sort_values(by="Differenza (Sofascore)", ascending=False).drop(columns=['xG Modello', 'Differenza (Modello)']).head(3)
    sofaOPdf = sofaOPdf.reset_index()
    modelOPdf = playersDF.sort_values(by="Differenza (Modello)", ascending=False).drop(columns=['xG Sofascore', 'Differenza (Sofascore)']).head(3)
    modelOPdf = modelOPdf.reset_index()
    with col1:
        st.write("Overperformer Secondo Sofascore")
        # st.dataframe(sofaOPdf, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaOPdf.loc[0]['ID Giocatore']) + "/image"
            p = sofaOPdf.loc[0]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Sofascore'], 2))
            pGoal = str(sofaOPdf.loc[0]['Goal'])
            pDiff = str(round(sofaOPdf.loc[0]['Differenza (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
        with c2:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaOPdf.loc[1]['ID Giocatore']) + "/image"
            p = sofaOPdf.loc[1]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Sofascore'], 2))
            pGoal = str(sofaOPdf.loc[1]['Goal'])
            pDiff = str(round(sofaOPdf.loc[1]['Differenza (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
        with c3:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaOPdf.loc[2]['ID Giocatore']) + "/image"
            p = sofaOPdf.loc[2]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Sofascore'], 2))
            pGoal = str(sofaOPdf.loc[2]['Goal'])
            pDiff = str(round(sofaOPdf.loc[2]['Differenza (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
    with col2:
        st.write("Overperformer Secondo il Modello")
        # st.dataframe(modelOPdf, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelOPdf.loc[0]['ID Giocatore']) + "/image"
            p = modelOPdf.loc[0]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Modello'], 2))
            pGoal = str(modelOPdf.loc[0]['Goal'])
            pDiff = str(round(modelOPdf.loc[0]['Differenza (Modello)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
        with c2:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelOPdf.loc[1]['ID Giocatore']) + "/image"
            p = modelOPdf.loc[1]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Modello'], 2))
            pGoal = str(modelOPdf.loc[1]['Goal'])
            pDiff = str(round(modelOPdf.loc[1]['Differenza (Modello)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
        with c3:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelOPdf.loc[2]['ID Giocatore']) + "/image"
            p = modelOPdf.loc[2]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Modello'], 2))
            pGoal = str(modelOPdf.loc[2]['Goal'])
            pDiff = str(round(modelOPdf.loc[2]['Differenza (Modello)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
    
    col1, col2 = st.columns(2)
    sofaUPdf = playersDF.sort_values(by="Differenza (Sofascore)", ascending=True).drop(columns=['xG Modello', 'Differenza (Modello)']).head(3)
    sofaUPdf = sofaUPdf.reset_index()
    modelUPdf = playersDF.sort_values(by="Differenza (Modello)", ascending=True).drop(columns=['xG Sofascore', 'Differenza (Sofascore)']).head(3)
    modelUPdf = modelUPdf.reset_index()
    with col1:
        st.write("Underperformer Secondo Sofascore")
        # st.dataframe(sofaUPdf, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaUPdf.loc[0]['ID Giocatore']) + "/image"
            p = sofaUPdf.loc[0]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Sofascore'], 2))
            pGoal = str(sofaUPdf.loc[0]['Goal'])
            pDiff = str(round(sofaUPdf.loc[0]['Differenza (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
        with c2:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaUPdf.loc[1]['ID Giocatore']) + "/image"
            p = sofaUPdf.loc[1]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Sofascore'], 2))
            pGoal = str(sofaUPdf.loc[1]['Goal'])
            pDiff = str(round(sofaUPdf.loc[1]['Differenza (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
        with c3:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaUPdf.loc[2]['ID Giocatore']) + "/image"
            p = sofaUPdf.loc[2]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Sofascore'], 2))
            pGoal = str(sofaUPdf.loc[2]['Goal'])
            pDiff = str(round(sofaUPdf.loc[2]['Differenza (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
    with col2:
        st.write("Underperformer Secondo il Modello")
        # st.dataframe(modelUPdf, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        
        
        
        with c1:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelUPdf.loc[0]['ID Giocatore']) + "/image"
            p = modelUPdf.loc[0]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Modello'], 2))
            pGoal = str(modelUPdf.loc[0]['Goal'])
            pDiff = str(round(modelUPdf.loc[0]['Differenza (Modello)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
        with c2:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelUPdf.loc[1]['ID Giocatore']) + "/image"
            p = modelUPdf.loc[1]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Modello'], 2))
            pGoal = str(modelUPdf.loc[1]['Goal'])
            pDiff = str(round(modelUPdf.loc[1]['Differenza (Modello)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
        with c3:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelUPdf.loc[2]['ID Giocatore']) + "/image"
            p = modelUPdf.loc[2]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Modello'], 2))
            pGoal = str(modelUPdf.loc[2]['Goal'])
            pDiff = str(round(modelUPdf.loc[2]['Differenza (Modello)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)


def photoKeepers(shotsDF):
    shotKeepers = np.unique(shotsDF['keeper'])
    keepers = []
    keeperIDs = []
    xgSums = []
    xgPredSums = []
    goalSums = []
    for keeper in shotKeepers:
        keeperShots = shotsDF.loc[shotsDF['keeper'] == keeper].reset_index()
        keeperID = keeperShots.loc[0]['keeperID']
        xgSum = np.sum(keeperShots['xg'])
        xgPredSum = np.sum(keeperShots['xgPred'])
        goalSum = np.sum(keeperShots['goal'])
        keepers.append(keeper)
        keeperIDs.append(keeperID)
        xgSums.append(xgSum)
        xgPredSums.append(xgPredSum)
        goalSums.append(goalSum)
    keepersDF = pd.DataFrame()
    keepersDF['Giocatore'] = keepers
    keepersDF['ID Giocatore'] = keeperIDs
    keepersDF['xG Sofascore'] = xgSums
    keepersDF['xG Modello'] = xgPredSums
    keepersDF['Goal'] = goalSums
    keepersDF['Differenza (Sofascore)'] = keepersDF['xG Sofascore'] - keepersDF['Goal']
    keepersDF['Differenza (Modello)'] = keepersDF['xG Modello'] - keepersDF['Goal']
    
    st.header("Portieri")

    col1, col2 = st.columns(2)
    sofaOPdf = keepersDF.sort_values(by="Differenza (Sofascore)", ascending=False).drop(columns=['xG Modello', 'Differenza (Modello)']).head(3)
    sofaOPdf = sofaOPdf.reset_index()
    modelOPdf = keepersDF.sort_values(by="Differenza (Modello)", ascending=False).drop(columns=['xG Sofascore', 'Differenza (Sofascore)']).head(3)
    modelOPdf = modelOPdf.reset_index()
    with col1:
        st.write("Overperformer Secondo Sofascore")
        
        # st.dataframe(sofaOPdf, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaOPdf.loc[0]['ID Giocatore']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = sofaOPdf.loc[0]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Sofascore'], 2))
            pGoal = str(sofaOPdf.loc[0]['Goal'])
            pDiff = str(round(sofaOPdf.loc[0]['Differenza (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (-" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
        with c2:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaOPdf.loc[1]['ID Giocatore']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = sofaOPdf.loc[1]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Sofascore'], 2))
            pGoal = str(sofaOPdf.loc[1]['Goal'])
            pDiff = str(round(sofaOPdf.loc[1]['Differenza (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (-" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
        with c3:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaOPdf.loc[2]['ID Giocatore']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = sofaOPdf.loc[2]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Sofascore'], 2))
            pGoal = str(sofaOPdf.loc[2]['Goal'])
            pDiff = str(round(sofaOPdf.loc[2]['Differenza (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (-" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
    with col2:
        st.write("Overperformer Secondo il Modello")
        # st.dataframe(modelOPdf, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelOPdf.loc[0]['ID Giocatore']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = modelOPdf.loc[0]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Modello'], 2))
            pGoal = str(modelOPdf.loc[0]['Goal'])
            pDiff = str(abs(round(modelOPdf.loc[0]['Differenza (Modello)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (-" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
        with c2:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelOPdf.loc[1]['ID Giocatore']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = modelOPdf.loc[1]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Modello'], 2))
            pGoal = str(modelOPdf.loc[1]['Goal'])
            pDiff = str(abs(round(modelOPdf.loc[1]['Differenza (Modello)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (-" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
        with c3:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelOPdf.loc[2]['ID Giocatore']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = modelOPdf.loc[2]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Modello'], 2))
            pGoal = str(modelOPdf.loc[2]['Goal'])
            pDiff = str(abs(round(modelOPdf.loc[2]['Differenza (Modello)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (-" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)

    col1, col2 = st.columns(2)
    sofaUPdf = keepersDF.sort_values(by="Differenza (Sofascore)", ascending=True).drop(columns=['xG Modello', 'Differenza (Modello)']).head(3)
    sofaUPdf = sofaUPdf.reset_index()
    modelUPdf = keepersDF.sort_values(by="Differenza (Modello)", ascending=True).drop(columns=['xG Sofascore', 'Differenza (Sofascore)']).head(3)
    modelUPdf = modelUPdf.reset_index()
    with col1:
        st.write("Underperformer Secondo Sofascore")
        # st.dataframe(sofaUPdf, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaUPdf.loc[0]['ID Giocatore']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = sofaUPdf.loc[0]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Sofascore'], 2))
            pGoal = str(sofaUPdf.loc[0]['Goal'])
            pDiff = str(abs(round(sofaUPdf.loc[0]['Differenza (Sofascore)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
        with c2:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaUPdf.loc[1]['ID Giocatore']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = sofaUPdf.loc[1]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Sofascore'], 2))
            pGoal = str(sofaUPdf.loc[1]['Goal'])
            pDiff = str(abs(round(sofaUPdf.loc[1]['Differenza (Sofascore)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
        with c3:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaUPdf.loc[2]['ID Giocatore']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = sofaUPdf.loc[2]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Sofascore'], 2))
            pGoal = str(sofaUPdf.loc[2]['Goal'])
            pDiff = str(abs(round(sofaUPdf.loc[2]['Differenza (Sofascore)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
    with col2:
        st.write("Underperformer Secondo il Modello")
        # st.dataframe(modelUPdf, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelUPdf.loc[0]['ID Giocatore']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = modelUPdf.loc[0]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Modello'], 2))
            pGoal = str(modelUPdf.loc[0]['Goal'])
            pDiff = str(abs(round(modelUPdf.loc[0]['Differenza (Modello)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
        with c2:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelUPdf.loc[1]['ID Giocatore']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = modelUPdf.loc[1]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Modello'], 2))
            pGoal = str(modelUPdf.loc[1]['Goal'])
            pDiff = str(abs(round(modelUPdf.loc[1]['Differenza (Modello)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)
        with c3:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelUPdf.loc[2]['ID Giocatore']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = modelUPdf.loc[2]
            pName = str(p['Giocatore'])
            pXG = str(round(p['xG Modello'], 2))
            pGoal = str(modelUPdf.loc[2]['Goal'])
            pDiff = str(abs(round(modelUPdf.loc[2]['Differenza (Modello)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, pName, pXG, pGoal, pDiff)



def displayCard(url, name, xg, goal, diff):
    card_html = f"""
    <div class="card">
        <img src="{url}" alt="Immagine della card">
        <div class="card-title">{name}</div>
        <div class="card-row">
            <div>xG: {xg}</div>
            <div>Goal: {goal}</div>
        </div>
        <div class="card-difference">Differenza: {diff}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


st.title("Serie A 2024/25")
st.subheader("Filtra per Partita e Tiro per vedere la mappa dei tiri e le differenze di xG!")

optionMenu = option_menu(None, ["Statistiche Tiri", "Statistiche Giocatori"],
    icons=['1-circle', '2-circle'],
    default_index=0, orientation="horizontal"
)
if optionMenu == "Statistiche Tiri":
    showShots()
elif optionMenu == "Statistiche Giocatori":
    showPlayers()
