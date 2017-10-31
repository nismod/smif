import fetch from 'isomorphic-fetch';


export const FETCH_PROJECTS = 'FETCH_PROJECTS';
function fetchProjects(){
    return {
        type: FETCH_PROJECTS
    };
}

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
