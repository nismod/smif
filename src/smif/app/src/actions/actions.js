import fetch from 'isomorphic-fetch'

export const REQUEST_SMIF_DETAILS = 'REQUEST_SMIF_DETAILS'
function requestSmifDetails(){
    return {
        type: REQUEST_SMIF_DETAILS
    }
}

export const RECEIVE_SMIF_DETAILS = 'RECEIVE_SMIF_DETAILS'
function receiveSmifDetails(json) {
    return {
        type: RECEIVE_SMIF_DETAILS,
        smif: json,
        receivedAt: Date.now()
    }
}

export function fetchSmifDetails(){
    return function (dispatch) {
        
        // inform the app that the API request is starting
        dispatch(requestSmifDetails())

        // make API request, returning a promise
        return fetch('/api/v1/smif/')
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                json => dispatch(receiveSmifDetails(json))
            )
    }
}

export const REQUEST_SOS_MODEL_RUNS = 'REQUEST_SOS_MODEL_RUNS'
function requestSosModelRuns(){
    return {
        type: REQUEST_SOS_MODEL_RUNS
    }
}

export const RECEIVE_SOS_MODEL_RUNS = 'RECEIVE_SOS_MODEL_RUNS'
function receiveSosModelRuns(json) {
    return {
        type: RECEIVE_SOS_MODEL_RUNS,
        sos_model_runs: json,
        receivedAt: Date.now()
    }
}

export function fetchSosModelRuns(){
    return function (dispatch) {

        // inform the app that the API request is starting
        dispatch(requestSosModelRuns())

        // make API request, returning a promise
        return fetch('/api/v1/sos_model_runs/')
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                json => dispatch(receiveSosModelRuns(json))
            )
    }
}

export const REQUEST_SOS_MODEL_RUN = 'REQUEST_SOS_MODEL_RUN'
function requestSosModelRun(){
    return {
        type: REQUEST_SOS_MODEL_RUN
    }
}

export const RECEIVE_SOS_MODEL_RUN = 'RECEIVE_SOS_MODEL_RUN'
function receiveSosModelRun(json) {
    return {
        type: RECEIVE_SOS_MODEL_RUN,
        sos_model_run: json,
        receivedAt: Date.now()
    }
}

export function fetchSosModelRun(modelrunid){
    return function (dispatch) {

        // inform the app that the API request is starting
        dispatch(requestSosModelRun())

        // make API request, returning a promise
        return fetch('/api/v1/sos_model_runs/' + modelrunid)
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                json => dispatch(receiveSosModelRun(json))
            )
    }
}

export function saveSosModelRun(modelrun){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/sos_model_runs/' + modelrun.name, {
            method: 'put',
            body: JSON.stringify(modelrun),

            headers: {
                'Content-Type': 'application/json'
            }}
        )
            .then (
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                //json => dispatch(receiveSosModelRun(json))
            )
    }
}

export function createSosModelRun(sosModelRun){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/sos_model_runs/', {
            method: 'post',
            body: JSON.stringify(sosModelRun),

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                data => dispatch(fetchSosModelRuns())
            )
    }
}

export function deleteSosModelRun(sosModelRunName){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/sos_model_runs/' + sosModelRunName, {
            method: 'delete',

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                data => dispatch(fetchSosModelRuns())
            )
    }
}

export const REQUEST_SOS_MODELS = 'REQUEST_SOS_MODELS'
function requestSosModels(){
    return {
        type: REQUEST_SOS_MODELS
    }
}

export const RECEIVE_SOS_MODELS = 'RECEIVE_SOS_MODELS'
function receiveSosModels(json) {
    return {
        type: RECEIVE_SOS_MODELS,
        sos_models: json,
        receivedAt: Date.now()
    }
}

export function fetchSosModels(){
    return function (dispatch) {

        // inform the app that the API request is starting
        dispatch(requestSosModels())

        // make API request, returning a promise
        return fetch('/api/v1/sos_models/')
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                json => dispatch(receiveSosModels(json))
            )
    }
}

export const REQUEST_SOS_MODEL = 'REQUEST_SOS_MODEL'
function requestSosModel(){
    return {
        type: REQUEST_SOS_MODEL
    }
}

export const RECEIVE_SOS_MODEL = 'RECEIVE_SOS_MODEL'
function receiveSosModel(json) {
    return {
        type: RECEIVE_SOS_MODEL,
        sos_model: json,
        receivedAt: Date.now()
    }
}

export function fetchSosModel(modelid){
    return function (dispatch) {

        // inform the app that the API request is starting
        dispatch(requestSosModel())

        // make API request, returning a promise
        return fetch('/api/v1/sos_models/' + modelid)
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                json => dispatch(receiveSosModel(json))
            )
    }
}

export function saveSosModel(model){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/sos_models/' + model.name, {
            method: 'put',
            body: JSON.stringify(model),

            headers: {
                'Content-Type': 'application/json'
            }}
        )
            .then (
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                //json => dispatch(receiveSosModelRun(json))
            )
    }
}

export function createSosModel(sosModel){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/sos_models/', {
            method: 'post',
            body: JSON.stringify(sosModel),

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                data => dispatch(fetchSosModels())
            )
    }
}

export function deleteSosModel(sosModelName){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/sos_models/' + sosModelName, {
            method: 'delete',

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                data => dispatch(fetchSosModels())
            )
    }
}

export const REQUEST_SECTOR_MODELS = 'REQUEST_SECTOR_MODELS'
function requestSectorModels(){
    return {
        type: REQUEST_SECTOR_MODELS
    }
}

export const RECEIVE_SECTOR_MODELS = 'RECEIVE_SECTOR_MODELS'
function receiveSectorModels(json) {
    return {
        type: RECEIVE_SECTOR_MODELS,
        sector_models: json,
        receivedAt: Date.now()
    }
}

export function fetchSectorModels(){
    return function (dispatch) {

        // inform the app that the API request is starting
        dispatch(requestSectorModels())

        // make API request, returning a promise
        return fetch('/api/v1/sector_models/')
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                json => dispatch(receiveSectorModels(json))
            )
    }
}

export const REQUEST_SECTOR_MODEL = 'REQUEST_SECTOR_MODEL'
function requestSectorModel(){
    return {
        type: REQUEST_SECTOR_MODEL
    }
}

export const RECEIVE_SECTOR_MODEL = 'RECEIVE_SECTOR_MODEL'
function receiveSectorModel(json) {
    return {
        type: RECEIVE_SECTOR_MODEL,
        sector_model: json,
        receivedAt: Date.now()
    }
}

export function fetchSectorModel(modelid){
    return function (dispatch) {

        // inform the app that the API request is starting
        dispatch(requestSectorModel())

        // make API request, returning a promise
        return fetch('/api/v1/sector_models/' + modelid)
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                json => dispatch(receiveSectorModel(json))
            )
    }
}

export function saveSectorModel(model){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/sector_models/' + model.name, {
            method: 'put',
            body: JSON.stringify(model),

            headers: {
                'Content-Type': 'application/json'
            }}
        )
            .then (
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                //json => dispatch(receiveSosModelRun(json))
            )
    }
}

export function createSectorModel(sectorModel){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/sector_models/', {
            method: 'post',
            body: JSON.stringify(sectorModel),

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                data => dispatch(fetchSectorModels())
            )
    }
}

export function deleteSectorModel(sectorModelName){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/sector_models/' + sectorModelName, {
            method: 'delete',

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                data => dispatch(fetchSectorModels())
            )
    }
}

export const REQUEST_SCENARIO_SETS = 'REQUEST_SCENARIO_SETS'
function requestScenarioSets(){
    return {
        type: REQUEST_SCENARIO_SETS
    }
}

export const RECEIVE_SCENARIO_SETS = 'RECEIVE_SCENARIO_SETS'
function receiveScenarioSets(json) {
    return {
        type: RECEIVE_SCENARIO_SETS,
        scenario_sets: json,
        receivedAt: Date.now()
    }
}

export function fetchScenarioSets(){
    return function (dispatch) {

        // inform the app that the API request is starting
        dispatch(requestScenarioSets())

        // make API request, returning a promise
        return fetch('/api/v1/scenario_sets/')
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                json => dispatch(receiveScenarioSets(json))
            )
    }
}

export const REQUEST_SCENARIO_SET = 'REQUEST_SCENARIO_SET'
function requestScenarioSet(){
    return {
        type: REQUEST_SCENARIO_SET
    }
}

export const RECEIVE_SCENARIO_SET = 'RECEIVE_SCENARIO_SET'
function receiveScenarioSet(json) {
    return {
        type: RECEIVE_SCENARIO_SET,
        scenario_set: json,
        receivedAt: Date.now()
    }
}

export function fetchScenarioSet(scenarioSetName){
    return function (dispatch) {

        // inform the app that the API request is starting
        dispatch(requestScenarioSet())

        // make API request, returning a promise
        return fetch('/api/v1/scenario_sets/' + scenarioSetName)
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                json => dispatch(receiveScenarioSet(json))
            )
    }
}

export function saveScenarioSet(scenarioSet){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/scenario_sets/' + scenarioSet.name, {
            method: 'put',
            body: JSON.stringify(scenarioSet),

            headers: {
                'Content-Type': 'application/json'
            }}
        )
            .then (
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                //json => dispatch(receiveSosModelRun(json))
            )
    }
}

export function createScenarioSet(scenarioSet){
    return function (dispatch) {
        // make API request, returning a promise
        return fetch('/api/v1/scenario_sets/', {
            method: 'post',
            body: JSON.stringify(scenarioSet),
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(
            response => response.json(),
            error => console.log('An error occurred.', error)
        )
        .then(
            data => dispatch(fetchScenarioSets())
        )
    }
}

export function deleteScenarioSet(scenarioSetName){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/scenario_sets/' + scenarioSetName, {
            method: 'delete',

            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(
            response => response.json(),
            error => console.log('An error occurred.', error)
        )
        .then(
            data => dispatch(fetchScenarioSets())
        )
    }
}

export const REQUEST_SCENARIOS = 'REQUEST_SCENARIOS'
function requestScenarios(){
    return {
        type: REQUEST_SCENARIOS
    }
}

export const RECEIVE_SCENARIOS = 'RECEIVE_SCENARIOS'
function receiveScenarios(json) {
    if (json == null) json = []
    return {
        type: RECEIVE_SCENARIOS,
        scenarios: json,
        receivedAt: Date.now()
    }
}

export function fetchScenarios(){
    return function (dispatch) {

        // inform the app that the API request is starting
        dispatch(requestScenarios())

        // make API request, returning a promise
        return fetch('/api/v1/scenarios/')
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                json => dispatch(receiveScenarios(json))
            )
    }
}

export const REQUEST_SCENARIO = 'REQUEST_SCENARIO'
function requestScenario(){
    return {
        type: REQUEST_SCENARIO
    }
}

export const RECEIVE_SCENARIO = 'RECEIVE_SCENARIO'
function receiveScenario(json) {
    return {
        type: RECEIVE_SCENARIO,
        scenario: json,
        receivedAt: Date.now()
    }
}

export function fetchScenario(scenarioid){
    return function (dispatch) {

        // inform the app that the API request is starting
        dispatch(requestScenario())

        // make API request, returning a promise
        return fetch('/api/v1/scenarios/' + scenarioid)
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                json => dispatch(receiveScenario(json))
            )
    }
}

export function saveScenario(scenario){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/scenarios/' + scenario.name, {
            method: 'put',
            body: JSON.stringify(scenario),

            headers: {
                'Content-Type': 'application/json'
            }}
        )
            .then (
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then (
                data => dispatch(fetchScenarios())
            )
    }
}

export function createScenario(scenario){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/scenarios/', {
            method: 'post',
            body: JSON.stringify(scenario),

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                data => dispatch(fetchScenarios())
            )
    }
}

export function deleteScenario(scenarioName){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/scenarios/' + scenarioName, {
            method: 'delete',

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                data => dispatch(fetchScenarios())
            )
    }
}

export const REQUEST_NARRATIVE_SETS = 'REQUEST_NARRATIVE_SETS'
function requestNarrativeSets(){
    return {
        type: REQUEST_NARRATIVE_SETS
    }
}

export const RECEIVE_NARRATIVE_SETS = 'RECEIVE_NARRATIVE_SETS'
function receiveNarrativeSets(json) {
    return {
        type: RECEIVE_NARRATIVE_SETS,
        narrative_sets: json,
        receivedAt: Date.now()
    }
}

export function fetchNarrativeSets(){
    return function (dispatch) {

        // inform the app that the API request is starting
        dispatch(requestNarrativeSets())

        // make API request, returning a promise
        return fetch('/api/v1/narrative_sets/')
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                json => dispatch(receiveNarrativeSets(json))
            )
    }
}

export const REQUEST_NARRATIVE_SET = 'REQUEST_NARRATIVE_SET'
function requestNarrativeSet(){
    return {
        type: REQUEST_NARRATIVE_SET
    }
}

export const RECEIVE_NARRATIVE_SET = 'RECEIVE_NARRATIVE_SET'
function receiveNarrativeSet(json) {
    return {
        type: RECEIVE_NARRATIVE_SET,
        narrative_set: json,
        receivedAt: Date.now()
    }
}

export function fetchNarrativeSet(narrativeSetName){
    return function (dispatch) {

        // inform the app that the API request is starting
        dispatch(requestNarrativeSet())

        // make API request, returning a promise
        return fetch('/api/v1/narrative_sets/' + narrativeSetName)
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                json => dispatch(receiveNarrativeSet(json))
            )
    }
}

export function saveNarrativeSet(narrativeSet){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/narrative_sets/' + narrativeSet.name, {
            method: 'put',
            body: JSON.stringify(narrativeSet),

            headers: {
                'Content-Type': 'application/json'
            }}
        )
            .then (
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                //json => dispatch(receiveSosModelRun(json))
            )
    }
}

export function createNarrativeSet(narrativeSet){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/narrative_sets/', {
            method: 'post',
            body: JSON.stringify(narrativeSet),

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                data => dispatch(fetchNarrativeSets())
            )
    }
}

export function deleteNarrativeSet(narrativeSetName){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/narrative_sets/' + narrativeSetName, {
            method: 'delete',

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                data => dispatch(fetchNarrativeSets())
            )
    }
}

export const REQUEST_NARRATIVES = 'REQUEST_NARRATIVES'
function requestNarratives(){
    return {
        type: REQUEST_NARRATIVES
    }
}

export const RECEIVE_NARRATIVES = 'RECEIVE_NARRATIVES'
function receiveNarratives(json) {
    return {
        type: RECEIVE_NARRATIVES,
        narratives: json,
        receivedAt: Date.now()
    }
}

export function fetchNarratives(){
    return function (dispatch) {
        // inform the app that the API request is starting
        dispatch(requestNarratives())

        // make API request, returning a promise
        return fetch('/api/v1/narratives/')
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                json => dispatch(receiveNarratives(json))
            )
    }
}

export const REQUEST_NARRATIVE = 'REQUEST_NARRATIVE'
function requestNarrative(){
    return {
        type: REQUEST_NARRATIVE
    }
}

export const RECEIVE_NARRATIVE = 'RECEIVE_NARRATIVE'
function receiveNarrative(json) {
    return {
        type: RECEIVE_NARRATIVE,
        narrative: json,
        receivedAt: Date.now()
    }
}

export function fetchNarrative(narrativeid){
    return function (dispatch) {
        // inform the app that the API request is starting
        dispatch(requestNarrative())

        // make API request, returning a promise
        return fetch('/api/v1/narratives/' + narrativeid)
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                json => dispatch(receiveNarrative(json))
            )
    }
}

export function saveNarrative(narrative){
    return function (dispatch) {
        // inform the app that the API request is starting

        // make API request, returning a promise
        return fetch('/api/v1/narratives/' + narrative.name, {
            method: 'put',
            body: JSON.stringify(narrative),

            headers: {
                'Content-Type': 'application/json'
            }}
        )
            .then (
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
    }
}

export function createNarrative(narrative){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/narratives/', {
            method: 'post',
            body: JSON.stringify(narrative),

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                data => dispatch(fetchNarratives())
            )
    }
}

export function deleteNarrative(narrativeName){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/narratives/' + narrativeName, {
            method: 'delete',

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                data => dispatch(fetchNarratives())
            )
    }
}
