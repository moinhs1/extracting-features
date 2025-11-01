#!/usr/bin/env python3
"""
Comprehensive Feature and Outcome Extraction from ALL Data Sources
Combines: Structured clinical data + Unstructured text data
Captures: Biomarkers, PE severity, Comorbidities, Interventions, Outcomes, and more
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveDataExtractor:
    def __init__(self):
        self.base_dir = Path("/home/moin/Research/TDA")
        self.rpdr_dir = self.base_dir / "rpdr_data"
        self.csv_dir = self.base_dir / "csv_final" 
        self.output_dir = self.base_dir / "all_features_outcomes"
        self.output_dir.mkdir(exist_ok=True)
        
        self.structured_sources = self._define_structured_sources()
        self.outcome_patterns = self._define_outcome_patterns()
        
    def _define_structured_sources(self):
        """Define structured data sources and their features"""
        sources = {
            'Lab': {
                'file_patterns': ['Lab.csv', 'Phy.csv', 'Phs.csv'],
                'features': {
                    # Cardiac biomarkers
                    'troponin': ['troponin', 'trop', 'ctni', 'ctnt'],
                    'bnp': ['bnp', 'brain natriuretic'],
                    'ntprobnp': ['nt-probnp', 'ntprobnp', 'nt probnp'],
                    'ddimer': ['d-dimer', 'ddimer', 'd dimer'],
                    
                    # Coagulation
                    'inr': ['inr', 'international normalized'],
                    'ptt': ['ptt', 'partial thromboplastin'],
                    'fibrinogen': ['fibrinogen'],
                    'platelet': ['platelet', 'plt'],
                    
                    # Renal function
                    'creatinine': ['creatinine', 'cr'],
                    'bun': ['bun', 'blood urea nitrogen'],
                    'egfr': ['egfr', 'gfr', 'glomerular'],
                    
                    # Liver function
                    'ast': ['ast', 'sgot'],
                    'alt': ['alt', 'sgpt'],
                    'bilirubin': ['bilirubin', 'bili'],
                    'albumin': ['albumin'],
                    
                    # Blood gas
                    'ph': ['ph', 'arterial ph'],
                    'pao2': ['pao2', 'po2', 'oxygen partial'],
                    'paco2': ['paco2', 'pco2', 'carbon dioxide'],
                    'lactate': ['lactate', 'lactic acid'],
                    
                    # Complete blood count
                    'hemoglobin': ['hemoglobin', 'hgb', 'hb'],
                    'wbc': ['wbc', 'white blood', 'white cell'],
                    'neutrophil': ['neutrophil', 'pmn', 'segs']
                }
            },
            'Dia': {
                'file_patterns': ['Dia.csv'],
                'features': {
                    # Comorbidities
                    'hypertension': ['i10', 'i11', 'i12', 'i13', 'hypertension'],
                    'diabetes': ['e10', 'e11', 'e08', 'diabetes'],
                    'heart_failure': ['i50', 'heart failure', 'chf'],
                    'cad': ['i25', 'i20', 'i21', 'coronary', 'cad'],
                    'copd': ['j44', 'copd', 'emphysema'],
                    'asthma': ['j45', 'asthma'],
                    'ckd': ['n18', 'chronic kidney', 'ckd'],
                    'cancer': ['c00-c97', 'malignant', 'cancer', 'carcinoma'],
                    'liver_disease': ['k70', 'k71', 'k72', 'k74', 'cirrhosis'],
                    'stroke': ['i63', 'i64', 'stroke', 'cva'],
                    'vte_history': ['i26', 'i80', 'i82', 'thrombosis', 'embolism'],
                    'obesity': ['e66', 'obesity', 'obese'],
                    'smoking': ['z87.891', 'f17', 'tobacco', 'smoking']
                }
            },
            'Med': {
                'file_patterns': ['Med.csv'],
                'features': {
                    # Anticoagulation
                    'heparin': ['heparin', 'ufh', 'unfractionated'],
                    'lovenox': ['enoxaparin', 'lovenox'],
                    'warfarin': ['warfarin', 'coumadin'],
                    'doac': ['apixaban', 'eliquis', 'rivaroxaban', 'xarelto', 'dabigatran', 'pradaxa'],
                    
                    # Thrombolysis
                    'tpa': ['alteplase', 'tpa', 'activase'],
                    
                    # Cardiac medications
                    'beta_blocker': ['metoprolol', 'carvedilol', 'bisoprolol', 'atenolol'],
                    'ace_arb': ['lisinopril', 'enalapril', 'losartan', 'valsartan'],
                    'diuretic': ['furosemide', 'lasix', 'bumetanide', 'torsemide'],
                    
                    # Pressors
                    'vasopressor': ['norepinephrine', 'epinephrine', 'vasopressin', 'dopamine', 'phenylephrine']
                }
            },
            'Phy': {
                'file_patterns': ['Phy.csv', 'Phs.csv'],
                'features': {
                    # Vital signs
                    'hypotension': ['sbp <90', 'hypotension', 'shock'],
                    'tachycardia': ['hr >100', 'tachycardia'],
                    'hypoxemia': ['o2 <90', 'hypoxemia', 'hypoxia'],
                    'fever': ['temp >38', 'fever', 'febrile']
                }
            }
        }
        return sources
    
    def _define_outcome_patterns(self):
        """Define comprehensive outcome patterns"""
        patterns = {
            # PE Severity Indicators
            'pe_severity': {
                'massive_pe': ['massive pe', 'saddle', 'high risk pe', 'submassive'],
                'bilateral_pe': ['bilateral', 'bilateral pe', 'bilateral pulmonary'],
                'rv_strain': ['rv strain', 'right ventricular strain', 'rv dysfunction'],
                'clot_burden': ['extensive', 'large burden', 'multiple segments', 'high clot burden']
            },
            
            # Biomarker Elevation
            'biomarkers': {
                'elevated_troponin': ['elevated troponin', 'troponin elevation', 'positive troponin'],
                'elevated_bnp': ['elevated bnp', 'bnp elevation', 'high bnp'],
                'elevated_ddimer': ['elevated d-dimer', 'd-dimer elevation', 'high d-dimer']
            },
            
            # Interventions
            'interventions': {
                'thrombolysis': ['tpa', 'alteplase', 'thrombolysis', 'thrombolytic'],
                'thrombectomy': ['thrombectomy', 'embolectomy', 'catheter directed'],
                'ivc_filter': ['ivc filter', 'inferior vena cava', 'filter placement'],
                'ecmo': ['ecmo', 'extracorporeal membrane'],
                'mechanical_support': ['impella', 'iabp', 'balloon pump']
            },
            
            # Bleeding Events
            'bleeding': {
                'major_bleeding': ['major bleeding', 'transfusion', 'prbc'],
                'gi_bleeding': ['gi bleed', 'melena', 'hematochezia'],
                'intracranial': ['intracranial', 'ich', 'subdural'],
                'minor_bleeding': ['minor bleeding', 'epistaxis', 'bruising']
            },
            
            # ICU and Critical Care
            'critical_care': {
                'icu_admission': ['icu', 'intensive care', 'critical care'],
                'intubation': ['intubat', 'mechanical ventilation', 'ventilator'],
                'vasopressors': ['vasopressor', 'norepinephrine', 'epinephrine'],
                'dialysis': ['dialysis', 'hemodialysis', 'crrt'],
                'cardiac_arrest': ['cardiac arrest', 'code blue', 'cpr']
            },
            
            # Mortality and Disposition
            'mortality': {
                'death': ['death', 'died', 'expired', 'deceased'],
                'hospice': ['hospice', 'palliative', 'comfort care'],
                'dnr': ['dnr', 'dni', 'do not resuscitate']
            },
            
            # Readmission
            'readmission': {
                '30day_readmit': ['readmission', 'readmit', 'return within 30'],
                'ed_visit': ['emergency department', 'ed visit', 'er visit']
            }
        }
        return patterns
    
    def load_structured_data(self):
        """Load and process structured data files"""
        print("\n" + "="*80)
        print("LOADING STRUCTURED DATA")
        print("="*80)
        
        structured_features = {}
        
        # Try to find structured data files
        possible_dirs = [self.rpdr_dir, self.base_dir / 'data', self.base_dir]
        
        for source_name, source_info in self.structured_sources.items():
            print(f"\nProcessing {source_name} data...")
            
            for pattern in source_info['file_patterns']:
                found = False
                for dir_path in possible_dirs:
                    file_path = dir_path / pattern
                    if file_path.exists():
                        print(f"  Found: {file_path}")
                        # Process file
                        try:
                            df = pd.read_csv(file_path, nrows=100000)  # Sample for speed
                            print(f"  Loaded {len(df)} records")
                            
                            # Extract features based on patterns
                            if source_name == 'Lab':
                                self.process_lab_data(df, structured_features)
                            elif source_name == 'Dia':
                                self.process_diagnosis_data(df, structured_features)
                            elif source_name == 'Med':
                                self.process_medication_data(df, structured_features)
                            
                            found = True
                            break
                        except Exception as e:
                            print(f"  Error processing {file_path}: {e}")
                
                if found:
                    break
        
        return structured_features
    
    def process_lab_data(self, df, features_dict):
        """Process laboratory data for biomarkers"""
        print("  Extracting laboratory features...")
        
        if 'EMPI' not in df.columns:
            print("    No EMPI column found")
            return
        
        # Group by patient
        for empi, patient_labs in df.groupby('EMPI'):
            empi = str(empi)
            if empi not in features_dict:
                features_dict[empi] = {}
            
            # Check for elevated biomarkers
            lab_text = ' '.join(patient_labs.fillna('').astype(str).values.flatten()).lower()
            
            # Troponin elevation
            if any(term in lab_text for term in ['troponin', 'trop']):
                # Look for numeric values
                trop_match = re.search(r'troponin[^\d]*(\d+\.?\d*)', lab_text)
                if trop_match:
                    value = float(trop_match.group(1))
                    features_dict[empi]['troponin_value'] = value
                    features_dict[empi]['elevated_troponin'] = int(value > 0.04)  # Common cutoff
            
            # BNP elevation
            if 'bnp' in lab_text:
                bnp_match = re.search(r'bnp[^\d]*(\d+)', lab_text)
                if bnp_match:
                    value = float(bnp_match.group(1))
                    features_dict[empi]['bnp_value'] = value
                    features_dict[empi]['elevated_bnp'] = int(value > 100)
            
            # D-dimer elevation
            if 'dimer' in lab_text:
                ddimer_match = re.search(r'd.?dimer[^\d]*(\d+\.?\d*)', lab_text)
                if ddimer_match:
                    value = float(ddimer_match.group(1))
                    features_dict[empi]['ddimer_value'] = value
                    features_dict[empi]['elevated_ddimer'] = int(value > 0.5)
            
            # Creatinine (kidney function)
            if 'creatinine' in lab_text:
                cr_match = re.search(r'creatinine[^\d]*(\d+\.?\d*)', lab_text)
                if cr_match:
                    value = float(cr_match.group(1))
                    features_dict[empi]['creatinine_value'] = value
                    features_dict[empi]['renal_dysfunction'] = int(value > 1.5)
    
    def process_diagnosis_data(self, df, features_dict):
        """Process diagnosis data for comorbidities"""
        print("  Extracting diagnosis/comorbidity features...")
        
        if 'EMPI' not in df.columns:
            print("    No EMPI column found")
            return
        
        # Group by patient
        for empi, patient_dx in df.groupby('EMPI'):
            empi = str(empi)
            if empi not in features_dict:
                features_dict[empi] = {}
            
            # Combine all diagnosis text
            dx_text = ' '.join(patient_dx.fillna('').astype(str).values.flatten()).lower()
            
            # Count comorbidities
            comorbidity_count = 0
            
            # Check each comorbidity
            for condition, terms in self.structured_sources['Dia']['features'].items():
                if any(term in dx_text for term in terms):
                    features_dict[empi][f'has_{condition}'] = 1
                    comorbidity_count += 1
                else:
                    features_dict[empi][f'has_{condition}'] = 0
            
            # Calculate comorbidity index
            features_dict[empi]['comorbidity_count'] = comorbidity_count
            features_dict[empi]['high_comorbidity'] = int(comorbidity_count >= 5)
            
            # Elixhauser approximation
            elixhauser_conditions = [
                'heart_failure', 'hypertension', 'diabetes', 'ckd', 'liver_disease',
                'copd', 'cancer', 'stroke'
            ]
            elixhauser_score = sum(1 for c in elixhauser_conditions if features_dict[empi].get(f'has_{c}', 0) == 1)
            features_dict[empi]['elixhauser_score'] = elixhauser_score
    
    def process_medication_data(self, df, features_dict):
        """Process medication data"""
        print("  Extracting medication features...")
        
        if 'EMPI' not in df.columns:
            print("    No EMPI column found")
            return
        
        # Group by patient
        for empi, patient_meds in df.groupby('EMPI'):
            empi = str(empi)
            if empi not in features_dict:
                features_dict[empi] = {}
            
            # Combine all medication text
            med_text = ' '.join(patient_meds.fillna('').astype(str).values.flatten()).lower()
            
            # Check medication categories
            for med_category, terms in self.structured_sources['Med']['features'].items():
                if any(term in med_text for term in terms):
                    features_dict[empi][f'received_{med_category}'] = 1
                else:
                    features_dict[empi][f'received_{med_category}'] = 0
            
            # Special flags
            if features_dict[empi].get('received_tpa', 0) == 1:
                features_dict[empi]['got_thrombolysis'] = 1
            
            if features_dict[empi].get('received_vasopressor', 0) == 1:
                features_dict[empi]['required_pressors'] = 1
    
    def extract_text_outcomes(self):
        """Extract outcomes from text data"""
        print("\n" + "="*80)
        print("EXTRACTING OUTCOMES FROM TEXT DATA")
        print("="*80)
        
        text_outcomes = {}
        
        # Process key text sources
        sources = ['Dis', 'Prg', 'Rad', 'Opn', 'Car']
        
        for source in sources:
            file_path = self.csv_dir / f"{source}.csv"
            if not file_path.exists():
                print(f"  {source} not found")
                continue
            
            print(f"\nProcessing {source}...")
            
            try:
                # Process in chunks
                chunk_count = 0
                for chunk in pd.read_csv(file_path, chunksize=5000):
                    chunk_count += 1
                    if chunk_count > 20:  # Limit processing
                        break
                    
                    if 'Report_Text' in chunk.columns and 'EMPI' in chunk.columns:
                        for _, row in chunk.iterrows():
                            empi = str(row['EMPI'])
                            text = str(row['Report_Text']).lower()
                            
                            if empi not in text_outcomes:
                                text_outcomes[empi] = {}
                            
                            # Extract outcomes from text
                            for category, patterns in self.outcome_patterns.items():
                                for outcome, terms in patterns.items():
                                    key = f'{category}_{outcome}'
                                    if any(term in text for term in terms):
                                        text_outcomes[empi][key] = 1
                            
                            # Extract PE severity indicators
                            if 'massive' in text or 'saddle' in text:
                                text_outcomes[empi]['severe_pe'] = 1
                            if 'bilateral' in text:
                                text_outcomes[empi]['bilateral_pe'] = 1
                            if 'rv strain' in text or 'right ventricular strain' in text:
                                text_outcomes[empi]['rv_strain'] = 1
                
            except Exception as e:
                print(f"  Error processing {source}: {e}")
        
        return text_outcomes
    
    def calculate_risk_scores(self, features_dict):
        """Calculate clinical risk scores"""
        print("\nCalculating risk scores...")
        
        for empi, features in features_dict.items():
            # PESI score approximation
            pesi_score = 0
            if features.get('age', 0) > 0:
                pesi_score += features['age']
            if features.get('sex', '') == 'M':
                pesi_score += 10
            if features.get('has_cancer', 0):
                pesi_score += 30
            if features.get('has_heart_failure', 0):
                pesi_score += 10
            if features.get('has_copd', 0):
                pesi_score += 10
            if features.get('tachycardia', 0):
                pesi_score += 20
            if features.get('hypotension', 0):
                pesi_score += 30
            if features.get('hypoxemia', 0):
                pesi_score += 20
            
            features['pesi_score'] = pesi_score
            features['high_risk_pesi'] = int(pesi_score > 85)
            
            # Simplified PESI
            spesi = 0
            if features.get('age', 0) > 80:
                spesi += 1
            if features.get('has_cancer', 0):
                spesi += 1
            if features.get('has_heart_failure', 0) or features.get('has_copd', 0):
                spesi += 1
            if features.get('tachycardia', 0):
                spesi += 1
            if features.get('hypotension', 0):
                spesi += 1
            if features.get('hypoxemia', 0):
                spesi += 1
            
            features['spesi_score'] = spesi
            features['high_risk_spesi'] = int(spesi >= 1)
    
    def merge_all_data(self, structured_features, text_outcomes):
        """Merge all data sources"""
        print("\n" + "="*80)
        print("MERGING ALL DATA SOURCES")
        print("="*80)
        
        # Combine dictionaries
        all_data = {}
        
        # Add structured features
        for empi, features in structured_features.items():
            all_data[empi] = features
        
        # Add text outcomes
        for empi, outcomes in text_outcomes.items():
            if empi not in all_data:
                all_data[empi] = {}
            all_data[empi].update(outcomes)
        
        # Calculate risk scores
        self.calculate_risk_scores(all_data)
        
        # Convert to dataframe
        df = pd.DataFrame.from_dict(all_data, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'EMPI'}, inplace=True)
        
        # Fill missing values
        df = df.fillna(0)
        
        # Create composite outcomes
        print("Creating composite outcomes...")
        
        # Severe PE composite
        severe_indicators = ['severe_pe', 'massive_pe', 'bilateral_pe', 'rv_strain', 
                           'elevated_troponin', 'elevated_bnp', 'hypotension']
        available = [c for c in severe_indicators if c in df.columns]
        if available:
            df['severe_pe_composite'] = df[available].max(axis=1)
        
        # High comorbidity composite
        if 'comorbidity_count' in df.columns:
            df['high_comorbidity_burden'] = (df['comorbidity_count'] >= 5).astype(int)
        
        # Critical illness composite
        critical_indicators = ['critical_care_icu_admission', 'critical_care_intubation',
                             'critical_care_vasopressors', 'critical_care_cardiac_arrest']
        available = [c for c in critical_indicators if c in df.columns]
        if available:
            df['critical_illness'] = df[available].max(axis=1)
        
        # Poor outcome composite
        poor_indicators = ['mortality_death', 'mortality_hospice', 'critical_illness',
                         'bleeding_major_bleeding', 'severe_pe_composite']
        available = [c for c in poor_indicators if c in df.columns]
        if available:
            df['poor_outcome'] = df[available].max(axis=1)
        
        print(f"Final dataset: {df.shape}")
        
        return df
    
    def generate_statistics(self, df):
        """Generate comprehensive statistics"""
        print("\n" + "="*80)
        print("COMPREHENSIVE STATISTICS")
        print("="*80)
        
        print("\nPatient Characteristics:")
        print(f"  Total patients: {len(df)}")
        
        if 'severe_pe_composite' in df.columns:
            print(f"  Severe PE: {df['severe_pe_composite'].sum()} ({df['severe_pe_composite'].mean()*100:.1f}%)")
        
        if 'elevated_troponin' in df.columns:
            print(f"  Elevated troponin: {df['elevated_troponin'].sum()} ({df['elevated_troponin'].mean()*100:.1f}%)")
        
        if 'elevated_bnp' in df.columns:
            print(f"  Elevated BNP: {df['elevated_bnp'].sum()} ({df['elevated_bnp'].mean()*100:.1f}%)")
        
        if 'comorbidity_count' in df.columns:
            print(f"  Mean comorbidities: {df['comorbidity_count'].mean():.1f}")
            print(f"  High comorbidity burden: {df['high_comorbidity_burden'].sum()} ({df['high_comorbidity_burden'].mean()*100:.1f}%)")
        
        print("\nInterventions:")
        intervention_cols = [c for c in df.columns if 'interventions_' in c or 'received_' in c]
        for col in intervention_cols[:10]:  # Top 10
            if df[col].sum() > 0:
                print(f"  {col}: {df[col].sum()} ({df[col].mean()*100:.1f}%)")
        
        print("\nOutcomes:")
        outcome_cols = [c for c in df.columns if 'mortality_' in c or 'bleeding_' in c or 'critical_' in c]
        for col in outcome_cols[:10]:  # Top 10
            if df[col].sum() > 0:
                print(f"  {col}: {df[col].sum()} ({df[col].mean()*100:.1f}%)")
        
        print("\nRisk Stratification:")
        if 'pesi_score' in df.columns:
            print(f"  Mean PESI: {df['pesi_score'].mean():.1f}")
            print(f"  High risk PESI: {df['high_risk_pesi'].sum()} ({df['high_risk_pesi'].mean()*100:.1f}%)")
        
        if 'spesi_score' in df.columns:
            print(f"  Mean sPESI: {df['spesi_score'].mean():.1f}")
            print(f"  High risk sPESI: {df['high_risk_spesi'].sum()} ({df['high_risk_spesi'].mean()*100:.1f}%)")
    
    def save_results(self, df):
        """Save all results"""
        print("\nSaving results...")
        
        # Save main dataframe
        df.to_csv(self.output_dir / 'all_features_outcomes.csv', index=False)
        print(f"  Saved: {self.output_dir / 'all_features_outcomes.csv'}")
        
        # Save feature list
        feature_list = {
            'clinical_features': [c for c in df.columns if not any(x in c for x in ['_', 'EMPI'])],
            'biomarker_features': [c for c in df.columns if any(x in c for x in ['troponin', 'bnp', 'ddimer'])],
            'comorbidity_features': [c for c in df.columns if 'has_' in c],
            'intervention_features': [c for c in df.columns if 'received_' in c or 'interventions_' in c],
            'outcome_features': [c for c in df.columns if any(x in c for x in ['mortality', 'bleeding', 'critical'])],
            'severity_features': [c for c in df.columns if 'severe' in c or 'risk' in c],
            'total_features': len(df.columns) - 1  # Exclude EMPI
        }
        
        with open(self.output_dir / 'feature_list.json', 'w') as f:
            json.dump(feature_list, f, indent=2)
        
        # Save summary statistics
        summary = {
            'timestamp': datetime.now().isoformat(),
            'n_patients': len(df),
            'n_features': len(df.columns) - 1,
            'severe_pe_rate': float(df['severe_pe_composite'].mean()) if 'severe_pe_composite' in df.columns else None,
            'mortality_rate': float(df['mortality_death'].mean()) if 'mortality_death' in df.columns else None,
            'intervention_rate': float(df[[c for c in df.columns if 'interventions_' in c]].max(axis=1).mean()) if any('interventions_' in c for c in df.columns) else None
        }
        
        with open(self.output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  Saved feature list and summary")
    
    def run_extraction(self):
        """Main execution"""
        start_time = datetime.now()
        
        # Load structured data
        structured_features = self.load_structured_data()
        
        # Extract text outcomes
        text_outcomes = self.extract_text_outcomes()
        
        # Merge all data
        final_df = self.merge_all_data(structured_features, text_outcomes)
        
        # Generate statistics
        self.generate_statistics(final_df)
        
        # Save results
        self.save_results(final_df)
        
        elapsed_time = datetime.now() - start_time
        print("\n" + "="*80)
        print("EXTRACTION COMPLETE")
        print("="*80)
        print(f"Total processing time: {elapsed_time}")
        print(f"Final dataset: {final_df.shape}")
        
        return final_df

def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE FEATURE AND OUTCOME EXTRACTION")
    print("Combining: Structured clinical data + Unstructured text data")
    print("="*80)
    
    extractor = ComprehensiveDataExtractor()
    df = extractor.run_extraction()
    
    print("\n✅ Extraction complete!")
    print("📁 Results saved in: all_features_outcomes/")
    print("\n🎯 Features Captured:")
    print("  • Cardiac biomarkers (troponin, BNP, D-dimer)")
    print("  • PE severity indicators")
    print("  • Comorbidity burden and indices")
    print("  • All interventions and procedures")
    print("  • Bleeding events")
    print("  • ICU and critical care")
    print("  • Mortality and readmissions")
    print("  • Risk scores (PESI, sPESI)")
    print("\nNext steps:")
    print("1. Run machine learning models on these features")
    print("2. Predict multiple outcomes")
    print("3. Identify phenotypes with different outcomes")

if __name__ == "__main__":
    main()