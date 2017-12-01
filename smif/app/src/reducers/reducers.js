import { combineReducers } from 'redux'
import {
    REQUEST_SOS_MODEL_RUNS,
    RECEIVE_SOS_MODEL_RUNS,
    REQUEST_SOS_MODEL_RUN,
    RECEIVE_SOS_MODEL_RUN,
    REQUEST_SOS_MODELS,
    RECEIVE_SOS_MODELS,
    REQUEST_SOS_MODEL,
    RECEIVE_SOS_MODEL,
    REQUEST_SECTOR_MODELS,
    RECEIVE_SECTOR_MODELS,
    REQUEST_SECTOR_MODEL,
    RECEIVE_SECTOR_MODEL,
    REQUEST_SCENARIO_SETS,
    RECEIVE_SCENARIO_SETS,
    REQUEST_SCENARIOS,
    RECEIVE_SCENARIOS,
    REQUEST_NARRATIVE_SETS,
    RECEIVE_NARRATIVE_SETS,
    REQUEST_NARRATIVES,
    RECEIVE_NARRATIVES,
} from '../actions/actions.js'

function sos_model_runs(
    state = {
        isFetching: false,
        items: []
    },
    action
) {
    switch (action.type){
    case REQUEST_SOS_MODEL_RUNS:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_SOS_MODEL_RUNS:
        return Object.assign({}, state, {
            isFetching: false,
            items: action.sos_model_runs,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function sos_model_run(
    state = {
        isFetching: false,
        item: {}
    },
    action
) {
    switch (action.type){
    case REQUEST_SOS_MODEL_RUN:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_SOS_MODEL_RUN:
        return Object.assign({}, state, {
            isFetching: false,
            item: action.sos_model_run,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function sos_models(
    state = {
        isFetching: false,
        items: []
    },
    action
) {
    switch (action.type){
    case REQUEST_SOS_MODELS:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_SOS_MODELS:
        return Object.assign({}, state, {
            isFetching: false,
            items: action.sos_models,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function sos_model(
    state = {
        isFetching: false,
        item: {}
    },
    action
) {
    switch (action.type){
    case REQUEST_SOS_MODEL:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_SOS_MODEL:
        return Object.assign({}, state, {
            isFetching: false,
            item: action.sos_model,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function sector_models(
    state = {
        isFetching: false,
        items: []
    },
    action
) {
    switch (action.type){
    case REQUEST_SECTOR_MODELS:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_SECTOR_MODELS:
        return Object.assign({}, state, {
            isFetching: false,
            items: action.sector_models,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function sector_model(
    state = {
        isFetching: false,
        item: {}
    },
    action
) {
    switch (action.type){
    case REQUEST_SECTOR_MODEL:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_SECTOR_MODEL:
        return Object.assign({}, state, {
            isFetching: false,
            item: action.sector_model,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function scenario_sets(
    state = {
        isFetching: false,
        items: []
    },
    action
) {
    switch (action.type){
    case REQUEST_SCENARIO_SETS:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_SCENARIO_SETS:
        return Object.assign({}, state, {
            isFetching: false,
            items: action.scenario_sets,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function scenarios(
    state = {
        isFetching: false,
        items: []
    },
    action
) {
    switch (action.type){
    case REQUEST_SCENARIOS:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_SCENARIOS:
        return Object.assign({}, state, {
            isFetching: false,
            items: action.scenarios,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function narrative_sets(
    state = {
        isFetching: false,
        items: []
    },
    action
) {
    switch (action.type){
    case REQUEST_NARRATIVE_SETS:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_NARRATIVE_SETS:
        return Object.assign({}, state, {
            isFetching: false,
            items: action.narrative_sets,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function narratives(
    state = {
        isFetching: false,
        items: []
    },
    action
) {
    switch (action.type){
    case REQUEST_NARRATIVES:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_NARRATIVES:
        return Object.assign({}, state, {
            isFetching: false,
            items: action.narratives,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

const rootReducer = combineReducers({
    sos_model_runs,
    sos_model_run,
    sos_models,
    sos_model,
    sector_models,
    sector_model,
    scenario_sets,
    scenarios,
    narrative_sets,
    narratives
})

export default rootReducer
