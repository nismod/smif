import fetch from 'isomorphic-fetch'

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
        // inform the app that the API request is starting

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

export function createSosModelRun(sosModelRunName){
    return function (dispatch) {
        // prepare the new modelrun
        let datetime = new Date()

        let newModelRun = {
            'name': sosModelRunName, 
            'description': '', 
            'stamp': datetime.toISOString(),  
            'sos_model': '',
            'scenarios': [],
            'narratives': [],
            'decision_module': '', 
            'timesteps': []
        }
        
        console.log(newModelRun)

        // make API request, returning a promise
        return fetch('/api/v1/sos_model_runs/', {
            method: 'post',
            body: JSON.stringify(newModelRun),

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
            //json => dispatch(receiveSosModelRun(json))
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
            //json => dispatch(receiveSosModelRun(json))
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

export const REQUEST_SCENARIOS = 'REQUEST_SCENARIOS'
function requestScenarios(){
    return {
        type: REQUEST_SCENARIOS
    }
}

export const RECEIVE_SCENARIOS = 'RECEIVE_SCENARIOS'
function receiveScenarios(json) {
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