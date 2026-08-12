import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests 

# 1. CONFIGURAREA PAGINII
st.set_page_config(page_title="F1 Teammate Predictor", page_icon="🏎️", layout="wide")
st.title("🏎️ F1 Teammate Battle Predictor")

# 2. INCARCAREA MODELELOR SI A DATELOR
@st.cache_resource
def load_assets():
    model = joblib.load('best_f1_model.pkl')
    scaler = joblib.load('scaler.pkl')
    imputer = joblib.load('imputer.pkl')
    df = pd.read_csv('f1_ready_data.csv')
    return model, scaler, imputer, df

model, scaler, imputer, df = load_assets()

# lista de features
FEATURES = [
    'grid', 'grid_advantage', 'age_advantage', 'quali_advantage', 'quali_time_advantage', 'champ_pts_advantage', 
    'champ_pos_advantage', 'wins_advantage', 'team_position', 'pit_stop_advantage', 'pit_time_advantage',
    'sprint_advantage', 'driver_exp', 'driver_win_rate', 'driver_finish_rate', 'h2h_advantage', 'season_form_advantage',
    'exp_advantage', 'win_rate_advantage', 'finish_rate_advantage', 'h2h_reliable',
    'recent_form_advantage', 'circuit_form_advantage'
]

# 3. INTERFATA CU TAB-URI
tab_live, tab_istoric = st.tabs(["🔴 Cursa Live (Următoarea)", "📚 Istoric Curse"])

# TAB-UL 1: CURSA LIVE
with tab_live:
    st.header("Predictii Live pentru Weekendul Curent")
    
    if 'live_data_fetched' not in st.session_state:
        st.session_state.live_data_fetched = False

    if st.button("Verifică statusul cursei următoare"):
        with st.spinner("Conectare la serverele F1..."):
            try:
                # API Call pt următoarea cursa
                url_next = "http://api.jolpi.ca/ergast/f1/current/next.json"
                resp_next = requests.get(url_next).json()
                
                race_data = resp_next['MRData']['RaceTable']['Races'][0]
                st.session_state.round_no = race_data['round']
                st.session_state.race_name = race_data['raceName']
                st.session_state.season = race_data['season']
                
                # API Call pt calificari
                url_quali = f"http://api.jolpi.ca/ergast/f1/{st.session_state.season}/{st.session_state.round_no}/qualifying.json"
                resp_quali = requests.get(url_quali).json()
                
                quali_results = resp_quali['MRData']['RaceTable']['Races']
                
                if len(quali_results) == 0:
                    st.session_state.quali_ready = False
                else:
                    st.session_state.quali_ready = True
                    st.session_state.quali_data = quali_results[0]['QualifyingResults']
                
                # Salvam faptul că am extras datele cu succes
                st.session_state.live_data_fetched = True
                
            except Exception as e:
                st.error(f"Eroare la preluarea datelor: {e}")

    # Daca datele au fost preluate (chiar daca pagina s-a reincarcat), afisam interfata
    if st.session_state.live_data_fetched:
        st.subheader(f"🏆 Următoarea cursă: {st.session_state.race_name} ({st.session_state.season}) - Runda {st.session_state.round_no}")
        
        if not st.session_state.quali_ready:
            st.warning("Nu sunt destule date! Calificările nu au avut loc. Revino după sesiunea de calificări!")
        else:
            st.success("Calificările s-au încheiat! Datele pentru grila de start au fost preluate.")
            
            df_live_quali = pd.json_normalize(st.session_state.quali_data)
            
            # Selectbox pentru echipe
            live_teams = df_live_quali['Constructor.name'].unique()
            team_options = ["Toate echipele"] + list(live_teams)
            selected_live_team = st.selectbox("Alege Echipa pentru predicția LIVE", team_options)
            
            teams_to_predict = live_teams if selected_live_team == "Toate echipele" else [selected_live_team]
            
            for current_team in teams_to_predict:
                team_drivers = df_live_quali[df_live_quali['Constructor.name'] == current_team]
                
                if len(team_drivers) == 2:
                    pilot_1_live = team_drivers.iloc[0]
                    pilot_2_live = team_drivers.iloc[1]
                    
                    def time_to_sec(t_str):
                        try:
                            parts = str(t_str).split(':')
                            return float(parts[0]) * 60 + float(parts[1])
                        except:
                            return np.nan
                    
                    p1_time = time_to_sec(pilot_1_live.get('Q3', pilot_1_live.get('Q2', pilot_1_live.get('Q1'))))
                    p2_time = time_to_sec(pilot_2_live.get('Q3', pilot_2_live.get('Q2', pilot_2_live.get('Q1'))))
                    
                    p1_grid = int(pilot_1_live['position'])
                    p2_grid = int(pilot_2_live['position'])
                    
                    p1_ref = pilot_1_live['Driver.driverId'].lower()
                    p2_ref = pilot_2_live['Driver.driverId'].lower()
                    
                    p1_history = df[df['driverRef'].str.contains(p1_ref, case=False, na=False)].sort_values('race_date', ascending=False).head(1)
                    
                    if not p1_history.empty:
                        p1_features = p1_history[FEATURES].copy()
                        
                        p1_features['grid'] = p1_grid
                        p1_features['grid_advantage'] = p2_grid - p1_grid
                        p1_features['quali_advantage'] = p2_grid - p1_grid
                        
                        if not np.isnan(p1_time) and not np.isnan(p2_time):
                            p1_features['quali_time_advantage'] = p2_time - p1_time
                        
                        p1_feat_scaled = scaler.transform(imputer.transform(p1_features))
                        prob_p1 = model.predict_proba(np.nan_to_num(p1_feat_scaled, nan=0.0))[0][1]
                        prob_p2 = 1 - prob_p1 
                        
                        st.markdown("---")
                        st.subheader(f"{current_team}")
                        colA, colB, colC = st.columns(3)
                        
                        with colA:
                            st.metric(label=f"{pilot_1_live['Driver.givenName']} {pilot_1_live['Driver.familyName']}", 
                                      value=f"{prob_p1:.1%}", 
                                      delta=f"Grid: P{p1_grid}", delta_color="off")
                            if prob_p1 > prob_p2:
                                st.success("FAVORIT 🏆")
                                
                        with colB:
                            st.markdown("<h3 style='text-align: center; margin-top: 15px;'>VS</h3>", unsafe_allow_html=True)
                            
                        with colC:
                            st.metric(label=f"{pilot_2_live['Driver.givenName']} {pilot_2_live['Driver.familyName']}", 
                                      value=f"{prob_p2:.1%}", 
                                      delta=f"Grid: P{p2_grid}", delta_color="off")
                            if prob_p2 > prob_p1:
                                st.success("FAVORIT 🏆")
                    else:
                        st.warning(f"Nu am găsit istoricul pentru {p1_ref} ({current_team}) în CSV-ul tău.")
                else:
                    if selected_live_team != "Toate echipele":
                        st.warning(f"{current_team} nu are 2 piloți care au participat la calificări.")

# TAB-UL 2: ISTORIC CURSE
with tab_istoric:
    st.header("Analizează curse din trecut")
    
    current_year = pd.Timestamp.now().year
    years = sorted(list(set(df['year'].unique()) | {current_year}), reverse=True)
    
    col1, col2, col3 = st.columns(3)
    selected_year = col1.selectbox("Alege Anul", years)
    
    @st.cache_data(ttl=3600)
    def get_completed_races(year):
        try:
            url = f"http://api.jolpi.ca/ergast/f1/{year}/results/1.json"
            resp = requests.get(url).json()
            races = resp['MRData']['RaceTable']['Races']
            return {r['raceName']: r['round'] for r in races}
        except Exception:
            return {}

    completed_races = get_completed_races(selected_year)
    
    if not completed_races:
        st.warning(f"Nu s-au putut prelua cursele pentru anul {selected_year} sau nu există curse finalizate încă.")
    else:
        race_names = list(completed_races.keys())
        selected_race = col2.selectbox("Alege Cursa", race_names)
        round_no = completed_races[selected_race]
        
        @st.cache_data(ttl=3600)
        def get_race_results(year, round_no):
            try:
                url = f"http://api.jolpi.ca/ergast/f1/{year}/{round_no}/results.json"
                resp = requests.get(url).json()
                return resp['MRData']['RaceTable']['Races'][0]['Results']
            except Exception:
                return []

        race_results = get_race_results(selected_year, round_no)
        
        if not race_results:
            st.warning("Nu s-au putut prelua rezultatele pentru această cursă.")
        else:
            df_results = pd.json_normalize(race_results)
            
            teams = df_results['Constructor.name'].unique()
            team_options = ["Toate echipele"] + list(teams)
            selected_team = col3.selectbox("Alege Echipa", team_options)
            
            if st.button("Generează Predicție", key="btn_istoric"):
                st.markdown(f"### Rezultate: {selected_race} ({selected_year})")
                
                teams_to_process = teams if selected_team == "Toate echipele" else [selected_team]
                
                for current_team in teams_to_process:
                    matchup_csv = df[(df['year'] == selected_year) & 
                                     (df['race_name'] == selected_race) & 
                                     (df['team_name'] == current_team)].copy()
                    
                    team_data_api = df_results[df_results['Constructor.name'] == current_team]
                    
                    if len(team_data_api) >= 2:
                        pilot_1_api = team_data_api.iloc[0]
                        pilot_2_api = team_data_api.iloc[1]
                        
                        p1_name = f"{pilot_1_api['Driver.givenName']} {pilot_1_api['Driver.familyName']}"
                        p2_name = f"{pilot_2_api['Driver.givenName']} {pilot_2_api['Driver.familyName']}"
                        
                        p1_pos = int(pilot_1_api.get('positionOrder', 99))
                        p2_pos = int(pilot_2_api.get('positionOrder', 99))
                        
                        p1_won_real = p1_pos < p2_pos
                        p2_won_real = p2_pos < p1_pos
                        
                        prob_1, prob_2 = 0.5, 0.5
                        
                        if len(matchup_csv) == 2:
                            features_data = matchup_csv[FEATURES].copy()
                            features_data = imputer.transform(features_data)
                            features_data = scaler.transform(features_data)
                            features_data = np.nan_to_num(features_data, nan=0.0)
                            
                            probs = model.predict_proba(features_data)[:, 1]
                            
                            p1_csv_ref = matchup_csv.iloc[0]['driverRef'].lower()
                            if p1_csv_ref in pilot_1_api['Driver.driverId'].lower():
                                prob_1, prob_2 = probs[0], probs[1]
                            else:
                                prob_1, prob_2 = probs[1], probs[0]
                                
                        else:
                            p1_ref = pilot_1_api['Driver.driverId'].lower()
                            p2_ref = pilot_2_api['Driver.driverId'].lower()
                            
                            p1_grid = int(pilot_1_api.get('grid', 0))
                            p2_grid = int(pilot_2_api.get('grid', 0))
                            
                            p1_history = df[df['driverRef'].str.contains(p1_ref, case=False, na=False)].sort_values('race_date', ascending=False).head(1)
                            
                            if not p1_history.empty:
                                p1_features = p1_history[FEATURES].copy()
                                p1_features['grid'] = p1_grid
                                p1_features['grid_advantage'] = p2_grid - p1_grid
                                p1_features['quali_advantage'] = p2_grid - p1_grid
                                
                                p1_feat_scaled = scaler.transform(imputer.transform(p1_features))
                                prob_1 = model.predict_proba(np.nan_to_num(p1_feat_scaled, nan=0.0))[0][1]
                                prob_2 = 1 - prob_1
                        
                        st.markdown("---")
                        st.subheader(f"{current_team}")
                        
                        colA, colB, colC = st.columns(3)
                        
                        with colA:
                            st.metric(label=p1_name, 
                                      value=f"{prob_1:.1%}",
                                      delta=f"Loc cursă: P{p1_pos}", delta_color="off")
                            if prob_1 > prob_2:
                                st.success("FAVORIT PREZIS")
                            if p1_won_real:
                                st.info("🏆 A CÂȘTIGAT ÎN REALITATE")
                        
                        with colB:
                            st.markdown("<h3 style='text-align: center; margin-top: 15px;'>VS</h3>", unsafe_allow_html=True)
                            
                        with colC:
                            st.metric(label=p2_name, 
                                      value=f"{prob_2:.1%}",
                                      delta=f"Loc cursă: P{p2_pos}", delta_color="off")
                            if prob_2 > prob_1:
                                st.success("FAVORIT PREZIS")
                            if p2_won_real:
                                st.info("🏆 A CÂȘTIGAT ÎN REALITATE")
                    else:
                        if selected_team != "Toate echipele":
                            st.warning(f"Echipa {current_team} nu are 2 piloți clasați în această cursă.")