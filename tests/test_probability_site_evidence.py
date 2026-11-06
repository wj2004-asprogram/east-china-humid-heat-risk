"""Scientific and delivery invariants for the fixed five-site translation."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import pytest

from src.probability_site_evidence import grids_for_scope, verify_subset, describe_predictions
from tools.probability_phase5_sites import read_json, sha


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/probability_phase5'


@pytest.fixture(scope='module')
def evidence():
    return (pd.read_csv(DATA/'site_mapping.csv'), pd.read_csv(DATA/'predictions.csv'),
            read_json(DATA/'site_cards.json')['cards'])


def test_masked_primary_is_missing_and_never_substituted(evidence):
    mapping, predictions, cards = evidence
    rejected = cards[0]
    assert rejected['historical'] is None
    assert rejected['model_summary'] == []
    assert rejected['mapping']['grid_index'] is None
    assert not rejected['mapping']['temporal_available']
    assert rejected['mapping']['primary_grid_id'] == '30.7500,121.7500'
    assert '31.0000,121.7500' not in predictions.primary_grid_id.values
    assert rejected['mapping']['accepted_partition_area_fraction'] == pytest.approx(.299467449059)


def test_shared_grid_and_buffer_do_not_invent_independent_or_transfer_evidence(evidence):
    mapping, predictions, cards = evidence
    assert len(grids_for_scope(mapping, 'temporal')) == 2
    assert len(grids_for_scope(mapping, 'strict')) == 1
    assert cards[1]['historical'] == cards[2]['historical'] == cards[3]['historical']
    assert cards[1]['model_summary'] == cards[2]['model_summary'] == cards[3]['model_summary']
    assert cards[4]['mapping']['spatial_role'] == 'buffer'
    assert {r['scope'] for r in cards[4]['model_summary']} == {'temporal'}
    physical=predictions[['primary_grid_id','target_date_bjt']].drop_duplicates()
    assert len(physical)==840 and len(predictions)==15120


def test_source_history_and_prediction_windows_are_distinct(evidence):
    _, predictions, cards=evidence
    history=pd.read_csv(DATA/'historical_daily.csv')
    assert len(history)==4600
    assert len(pd.read_csv(DATA/'historical_annual.csv'))==50
    assert cards[1]['historical']['period']['days_per_summer']==92
    assert set(pd.to_datetime(history.date).dt.strftime('%m-%d')) >= {'06-01','08-31'}
    assert set(pd.to_datetime(predictions.target_date_bjt).dt.strftime('%m-%d')) == set(pd.date_range('2020-06-09','2020-08-31').strftime('%m-%d'))
    for lag,group in predictions.groupby('lag'):
        end=pd.to_datetime(group.max_input_valid_time_bjt,utc=True)
        target=pd.to_datetime(group.target_start_bjt,utc=True)
        assert (target-end==pd.Timedelta(days=int(lag)-1,hours=1)).all()


def test_unrounded_labels_common_samples_and_alerts_reject_tampering(evidence):
    _, predictions,_=evidence
    part=predictions[(predictions.scope=='strict')&(predictions.model_id=='logistic_F4')].copy()
    history=pd.read_csv(DATA/'historical_daily.csv').to_dict('records')
    kwargs=dict(historical_daily=history,years=[2020,2021,2022,2023,2024],expected_days=84,threshold=40.6)
    assert verify_subset(part,**kwargs)['distinct_physical_grid_days']==420
    bad=part.copy(); bad.loc[bad.index[0],'y']=1-bad.iloc[0].y
    with pytest.raises(ValueError,match='different labels|unrounded HI'):
        verify_subset(bad,**kwargs)
    bad=part.copy(); bad.loc[bad.index[0],'alert_budget']=1-bad.iloc[0].alert_budget
    with pytest.raises(ValueError,match='Frozen alert'):
        verify_subset(bad,**kwargs)
    with pytest.raises(ValueError,match='Duplicated'):
        verify_subset(pd.concat([part,part.iloc[:1]]),**kwargs)
    with pytest.raises(ValueError,match='Incomplete prediction summer'):
        verify_subset(part.iloc[1:],**kwargs)


def test_annual_brier_and_onset_denominators_match_direct_calculation(evidence):
    _,predictions,_=evidence
    aggregate=pd.read_csv(DATA/'aggregate_scores.csv')
    for grid,n,events in [('30.2500,120.5000',157,40),('31.0000,121.5000',210,44)]:
        sub=predictions[(predictions.scope=='temporal')&(predictions.lag==2)&
                        (predictions.model_id=='logistic_F4')&(predictions.primary_grid_id==grid)&
                        (predictions.evaluation_previous_y==0)]
        row=aggregate[(aggregate.scope=='temporal')&(aggregate.lag==2)&
                      (aggregate.model_id=='logistic_F4')&(aggregate.primary_grid_id==grid)&
                      (aggregate.risk_set=='onset')&(aggregate.probability=='raw')].iloc[0]
        direct=sub.assign(loss=(sub.p_raw-sub.y)**2).groupby('year').loss.mean().mean()
        assert row.brier==pytest.approx(direct,abs=1e-14)
        assert row.n==len(sub)==n and row.events==sub.y.sum()==events


def test_sparse_risk_set_keeps_undefined_ap_and_recall():
    frame=pd.DataFrame(dict(scope=['temporal']*2,lag=[2]*2,model_id=['fixed']*2,
                            primary_grid_id=['g']*2,year=[2020]*2,y=[0,0],
                            evaluation_previous_y=[0,0],grid_latitude=[30,30],
                            p_raw=[.1,.2],p_calibrated=[.1,.2],
                            primary_threshold=[.5,.5],budget_threshold=[1.000001]*2))
    result=describe_predictions(frame,['all','onset','persistence_or_clearance'])
    assert result['aggregate_scores'].ap.isna().all()
    empty=result['annual_scores'][result['annual_scores'].risk_set=='persistence_or_clearance']
    assert (empty.n==0).all() and empty.brier.isna().all()
    assert result['annual_alerts'].recall.isna().all()


def test_missing_onsite_information_is_null_and_suggestions_remain_conditional(evidence):
    for card in evidence[2]:
        assert len(card['unknowns'])==13 and all(v is None for v in card['unknowns'].values())
        assert '待验证建议' in card['notes']['next_action']
        assert all(p['status']=='合理但未验证路径' for p in card['impact_paths'])
        assert card['snapshot_date']=='2026-10-29'


def test_extracted_outputs_and_report_hashes_are_exact():
    for filename in ['extraction_manifest.json','report_manifest.json']:
        manifest=read_json(DATA/filename)
        for path,expected in manifest['outputs_sha256'].items():
            assert sha(ROOT/path)==expected,path
        for path,expected in manifest.get('source_code_sha256',{}).items():
            assert sha(ROOT/path)==expected,path
    report=read_json(DATA/'report_manifest.json')
    assert report['source_extraction_manifest_sha256']==sha(DATA/'extraction_manifest.json')


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[];self.ids=[];self.scripts=0
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='script': self.scripts+=1
        if 'id' in attrs: self.ids.append(attrs['id'])
        for k in ('href','src'):
            if k in attrs:self.links.append(attrs[k])


def test_static_report_links_images_and_anchors_are_local_and_resolve():
    path=ROOT/'reports/probability_phase5/index.html'
    parser=Links();parser.feed(path.read_text(encoding='utf-8'))
    assert not parser.scripts and len(parser.ids)==5 and len(set(parser.ids))==5
    for link in parser.links:
        parsed=urlparse(link)
        assert not parsed.scheme,link
        if parsed.path: assert (path.parent/unquote(parsed.path)).resolve().is_file(),link
        if parsed.fragment: assert parsed.fragment in parser.ids,link


def test_card_source_ids_urls_and_hash_metadata_match_archived_catalog(evidence):
    catalog=read_json(DATA/'source_catalog.json')
    for card in evidence[2]:
        for source in card['evidence_ids']:
            row=catalog['sources'][source]
            assert card['source_urls'][source]==row['final_url']
            assert len(row['sha256'])==64 and row['retrieved_at_utc'].startswith('2026-10-29')


def test_model_candidates_and_grid_calendar_coverage_match_protocol(evidence):
    config=read_json(ROOT/'configs/probability_phase5_sites.json')
    prediction=evidence[1]
    assert set(prediction.model_id)==set(config['model_ids'])
    assert set(prediction.lag)==set(config['lags'])
    assert set(prediction.year)==set(config['final_years'])
    assert (prediction.groupby(['scope','lag','model_id','year','primary_grid_id']).size()==84).all()
    assert len(pd.read_csv(DATA/'annual_scores.csv'))==1080
    assert len(pd.read_csv(DATA/'reliability.csv'))==10800


def test_new_markdown_delivery_links_resolve():
    paths=list((ROOT/'docs').glob('PROBABILITY_PHASE5_*.md'))+list((ROOT/'reports/probability_phase5/cards').glob('*.md'))
    for path in paths:
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            parsed=urlparse(target)
            if not parsed.scheme and parsed.path:
                assert (path.parent/unquote(parsed.path)).resolve().exists(),(path,target)
