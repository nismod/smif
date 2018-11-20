import fetch from 'isomorphic-fetch'

export const APP_FORM_EDIT = 'APP_FORM_EDIT'
export function setAppFormEdit(){
    return function (dispatch) {
        dispatch({type: APP_FORM_EDIT})
    }
}

export const APP_FORM_SAVE = 'APP_FORM_SAVE'
export function setAppFormSave(){
    return function (dispatch) {
        dispatch({type: APP_FORM_SAVE})
    }
}

export const APP_FORM_DISCARD = 'APP_FORM_DISCARD'
export function setAppFormDiscard(){
    return function (dispatch) {
        dispatch({type: APP_FORM_DISCARD})
    }
}

export const APP_FORM_KEEP_EDITING = 'APP_FORM_KEEP_EDITING'
export function setAppFormKeepEditing(){
    return function (dispatch) {
        dispatch({type: APP_FORM_KEEP_EDITING})
    }
}

export const APP_FORM_SAVE_DONE = 'APP_FORM_SAVE_DONE'
export function setAppFormSaveDone(){
    return function (dispatch) {
        dispatch({type: APP_FORM_SAVE_DONE})
    }
}

export const APP_FORM_ACCEPT = 'APP_FORM_ACCEPT'
export function setAppFormAccept(){
    return function (dispatch) {
        dispatch({type: APP_FORM_ACCEPT})
    }
}

export const APP_FORM_REJECT = 'APP_FORM_REJECT'
export function setAppFormReject(){
    return function (dispatch) {
        dispatch({type: APP_FORM_REJECT})
    }
}

export const APP_NAVIGATE = 'APP_NAVIGATE'
export function setAppNavigate(url){
    return function (dispatch) {
        dispatch({
            type: APP_NAVIGATE,
            data: url
        })
    }
}

export const APP_NAVIGATE_DONE = 'APP_NAVIGATE_DONE'
export function setAppNavigateDone(url){
    return function (dispatch) {
        dispatch({
            type: APP_NAVIGATE_DONE,
            data: url
        })
    }
}

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
        data: json['data'],
        error: json['error'],
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
                response => response.json()
            )
            .then(
                json => dispatch(receiveSmifDetails(json))
            )
    }
}

export const REQUEST_MODEL_RUNS = 'REQUEST_MODEL_RUNS'
function requestModelRuns(){
    return {
        type: REQUEST_MODEL_RUNS
    }
}

export const RECEIVE_MODEL_RUNS = 'RECEIVE_MODEL_RUNS'
function receiveModelRuns(json) {
    return {
        type: RECEIVE_MODEL_RUNS,
        data: json['data'],
        error: json['error'],
        receivedAt: Date.now()
    }
}

export function fetchModelRuns(filter = undefined){
    return function (dispatch) {

        // inform the app that the API request is starting
        dispatch(requestModelRuns())

        // make API request, returning a promise
        if (filter == undefined) {
            return fetch('/api/v1/model_runs/')
                .then(
                    response => response.json()
                )
                .then(
                    json => dispatch(receiveModelRuns(json))
                )
        } else {
            return fetch('/api/v1/model_runs/?' + filter)
                .then(
                    response => response.json()
                )
                .then(
                    json => dispatch(receiveModelRuns(json))
                )
        }
    }
}

export const REQUEST_MODEL_RUN = 'REQUEST_MODEL_RUN'
function requestModelRun(){
    return {
        type: REQUEST_MODEL_RUN
    }
}

export const RECEIVE_MODEL_RUN = 'RECEIVE_MODEL_RUN'
function receiveModelRun(json) {
    return {
        type: RECEIVE_MODEL_RUN,
        data: json['data'],
        error: json['error'],
        receivedAt: Date.now()
    }
}

export function fetchModelRun(modelrunid){
    return function (dispatch) {

        // inform the app that the API request is starting
        dispatch(requestModelRun())

        // make API request, returning a promise
        return fetch('/api/v1/model_runs/' + modelrunid)
            .then(
                response => response.json()
            )
            .then(
                json => dispatch(receiveModelRun(json))
            )
    }
}

export const REQUEST_MODEL_RUN_STATUS = 'REQUEST_MODEL_RUN_STATUS'
function requestModelRunStatus(){
    return {
        type: REQUEST_MODEL_RUN_STATUS
    }
}

export const RECEIVE_MODEL_RUN_STATUS = 'RECEIVE_MODEL_RUN_STATUS'
function receiveModelRunStatus(json) {
    return {
        type: RECEIVE_MODEL_RUN_STATUS,
        data: json['data'],
        error: json['error'],
        receivedAt: Date.now()
    }
}

export function fetchModelRunStatus(modelrunid){
    return function (dispatch) {

        // inform the app that the API request is starting
        dispatch(requestModelRunStatus())

        // make API request, returning a promise
        return fetch('/api/v1/model_runs/' + modelrunid + '/status')
            .then(
                response => response.json()
            )
            .then(
                json => dispatch(receiveModelRunStatus(json))
            )
    }
}

export function saveModelRun(modelrun){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/model_runs/' + modelrun.name, {
            method: 'put',
            body: JSON.stringify(modelrun),

            headers: {
                'Content-Type': 'application/json'
            }}
        )
            .then (
                response => response.json()
            )
            .then (
                dispatch(setAppFormAccept())
            )
    }
}

export function createModelRun(ModelRun){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/model_runs/', {
            method: 'post',
            body: JSON.stringify(ModelRun),

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                function() {
                    response => response.json()
                    dispatch(fetchModelRuns())
                }
            )
    }
}

export function deleteModelRun(ModelRunName){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/model_runs/' + ModelRunName, {
            method: 'delete',

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                function() {
                    response => response.json()
                    dispatch(fetchModelRuns())
                }
            )
    }
}

export function startModelRun(ModelRunName, args){
    return function () {

        // make API request, returning a promise
        return fetch('/api/v1/model_runs/' + ModelRunName + '/start', {
            method: 'post',
            body: JSON.stringify({
                args: args
            }),

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                response => response.json()
            )
    }
}

export function killModelRun(ModelRunName){
    return function () {

        // make API request, returning a promise
        return fetch('/api/v1/model_runs/' + ModelRunName + '/kill', {
            method: 'post',

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                response => response.json()
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
        data: json['data'],
        error: json['error'],
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
                response => response.json()
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
        data: json['data'],
        error: json['error'],
        receivedAt: Date.now()
    }
}

export const SEND_SOS_MODEL = 'SEND_SOS_MODEL'
function sendSosModel(){
    return {
        type: SEND_SOS_MODEL
    }
}

export const REJECT_SOS_MODEL = 'REJECT_SOS_MODEL'
function rejectSosModel(json){
    return {
        type: REJECT_SOS_MODEL,
        data: json['data'],
        error: json['error']
    }
}

export const ACCEPT_SOS_MODEL = 'ACCEPT_SOS_MODEL'
export function acceptSosModel(){
    return {
        type: ACCEPT_SOS_MODEL,
        error: {}
    }
}

export function fetchSosModel(modelid){
    return function (dispatch) {

        // inform the app that the API request is starting
        dispatch(requestSosModel())

        // make API request, returning a promise
        return fetch('/api/v1/sos_models/' + modelid)
            .then(
                response => response.json()
            )
            .then(
                json => dispatch(receiveSosModel(json))
            )
    }
}

export function saveSosModel(model){
    return function (dispatch) {

        // inform the app that the API is saving
        dispatch(sendSosModel())

        // make API request, returning a promise
        return fetch('/api/v1/sos_models/' + model.name, {
            method: 'put',
            body: JSON.stringify(model),

            headers: {
                'Content-Type': 'application/json'
            }})
            .then(
                response => response.json()
            )
            .then(
                function(json) {
                    if (json['message'] == 'failed') {
                        dispatch(rejectSosModel(json))
                        dispatch(setAppFormReject())
                    }
                    else {
                        dispatch(acceptSosModel())
                        dispatch(setAppFormAccept())
                    }
                }
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
                function() {
                    response => response.json()
                    dispatch(fetchSosModels())
                }
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
                function() {
                    response => response.json()
                    dispatch(fetchSosModels())
                }
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
        data: json['data'],
        error: json['error'],
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
                response => response.json()
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
        data: json['data'],
        error: json['error'],
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
                response => response.json()
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
                response => response.json()
            )
            .then (
                dispatch(setAppFormAccept())
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
                function() {
                    response => response.json()
                    dispatch(fetchSectorModels())
                }
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
                function() {
                    response => response.json()
                    dispatch(fetchSectorModels())
                }
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
        data: json['data'],
        error: json['error'],
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
                response => response.json()
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
        data: json['data'],
        error: json['error'],
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
                response => response.json()
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
                function() {
                    response => response.json()
                    dispatch(fetchScenarios())
                }
            )
            .then (
                dispatch(setAppFormAccept())
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
                function() {
                    response => response.json()
                    dispatch(fetchScenarios())
                }
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
                function() {
                    response => response.json()
                    dispatch(fetchScenarios())
                }
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
        data: json['data'],
        error: json['error'],
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
                response => response.json()
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
        data: json['data'],
        error: json['error'],
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
                response => response.json()
            )
            .then(
                json => dispatch(receiveNarrative(json))
            )
    }
}

export function saveNarrative(narrative){
    return function () {
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
                response => response.json()
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
                function() {
                    response => response.json()
                    dispatch(fetchNarratives())
                }
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
                function() {
                    response => response.json()
                    dispatch(fetchNarratives())
                }
            )
    }
}

export const REQUEST_DIMENSIONS = 'REQUEST_DIMENSIONS'
function requestDimensions(){
    return {
        type: REQUEST_DIMENSIONS
    }
}

export const RECEIVE_DIMENSIONS = 'RECEIVE_DIMENSIONS'
function receiveDimensions(json) {
    return {
        type: RECEIVE_DIMENSIONS,
        data: json['data'],
        error: json['error'],
        receivedAt: Date.now()
    }
}

export function fetchDimensions(){
    return function (dispatch) {
        // inform the app that the API request is starting
        dispatch(requestDimensions())

        // make API request, returning a promise
        return fetch('/api/v1/dimensions/')
            .then(
                response => response.json()
            )
            .then(
                json => dispatch(receiveDimensions(json))
            )
    }
}

export const REQUEST_DIMENSION = 'REQUEST_DIMENSION'
function requestDimension(){
    return {
        type: REQUEST_DIMENSION
    }
}

export const RECEIVE_DIMENSION = 'RECEIVE_DIMENSION'
function receiveDimension(json) {
    return {
        type: RECEIVE_DIMENSION,
        data: json['data'],
        error: json['error'],
        receivedAt: Date.now()
    }
}

export function fetchDimension(dimension){
    return function (dispatch) {
        // inform the app that the API request is starting
        dispatch(requestDimension())

        // make API request, returning a promise
        return fetch('/api/v1/dimensions/' + dimension)
            .then(
                response => response.json()
            )
            .then(
                json => dispatch(receiveDimension(json))
            )
    }
}

export function saveDimension(dimension){
    return function () {
        // inform the app that the API request is starting

        // make API request, returning a promise
        return fetch('/api/v1/dimensions/' + dimension.name, {
            method: 'put',
            body: JSON.stringify(dimension),

            headers: {
                'Content-Type': 'application/json'
            }}
        )
            .then (
                response => response.json()
            )
    }
}

export function createDimension(dimension){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/dimensions/', {
            method: 'post',
            body: JSON.stringify(dimension),

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                function() {
                    response => response.json()
                    dispatch(fetchDimensions())
                }
            )
    }
}

export function deleteDimension(dimension){
    return function (dispatch) {

        // make API request, returning a promise
        return fetch('/api/v1/dimensions/' + dimension, {
            method: 'delete',

            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(
                function() {
                    response => response.json()
                    dispatch(fetchDimensions())
                }
            )
    }
}
