import fetch from 'isomorphic-fetch';

export const REQUEST_SOS_MODEL_RUNS = 'REQUEST_SOS_MODEL_RUNS';
function requestSosModelRuns(){
    return {
        type: REQUEST_SOS_MODEL_RUNS
    };
}

export const RECEIVE_SOS_MODEL_RUNS = 'RECEIVE_SOS_MODEL_RUNS';
function receiveSosModelRuns(json) {
    return {
        type: RECEIVE_SOS_MODEL_RUNS,
        sos_model_runs: json,
        receivedAt: Date.now()
    };
}

export function fetchSosModelRuns(){
    return function (dispatch) {
        // inform the app that the API request is starting
        dispatch(requestSosModelRuns());

        // make API request, returning a promise
        return fetch('/api/v1/sos_model_runs/')
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                json => dispatch(receiveSosModelRuns(json))
            );
    };
}

export const REQUEST_SOS_MODEL_RUN = 'REQUEST_SOS_MODEL_RUN';
function requestSosModelRun(){
    return {
        type: REQUEST_SOS_MODEL_RUN
    };
}

export const RECEIVE_SOS_MODEL_RUN = 'RECEIVE_SOS_MODEL_RUN';
function receiveSosModelRun(json) {
    return {
        type: RECEIVE_SOS_MODEL_RUN,
        sos_model_run: json,
        receivedAt: Date.now()
    };
}

export function fetchSosModelRun(modelrunid){
    return function (dispatch) {
        // inform the app that the API request is starting
        dispatch(requestSosModelRun());

        // make API request, returning a promise
        return fetch('/api/v1/sos_model_runs/' + modelrunid)
            .then(
                response => response.json(),
                error => console.log('An error occurred.', error)
            )
            .then(
                json => dispatch(receiveSosModelRun(json))
            );
    };
}