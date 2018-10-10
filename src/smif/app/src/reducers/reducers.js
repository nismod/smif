import { combineReducers } from 'redux'
import {
    REQUEST_SMIF_DETAILS,
    RECEIVE_SMIF_DETAILS,
    REQUEST_MODEL_RUNS,
    RECEIVE_MODEL_RUNS,
    REQUEST_MODEL_RUN,
    RECEIVE_MODEL_RUN,
    REQUEST_MODEL_RUN_STATUS,
    RECEIVE_MODEL_RUN_STATUS,
    REQUEST_SOS_MODELS,
    RECEIVE_SOS_MODELS,
    REQUEST_SOS_MODEL,
    RECEIVE_SOS_MODEL,
    SEND_SOS_MODEL,
    ACCEPT_SOS_MODEL,
    REJECT_SOS_MODEL,
    REQUEST_SECTOR_MODELS,
    RECEIVE_SECTOR_MODELS,
    REQUEST_SECTOR_MODEL,
    RECEIVE_SECTOR_MODEL,
    REQUEST_SCENARIOS,
    RECEIVE_SCENARIOS,
    REQUEST_SCENARIO,
    RECEIVE_SCENARIO,
    REQUEST_NARRATIVES,
    RECEIVE_NARRATIVES,
    REQUEST_NARRATIVE,
    RECEIVE_NARRATIVE,
    REQUEST_DIMENSIONS,
    RECEIVE_DIMENSIONS,
    REQUEST_DIMENSION,
    RECEIVE_DIMENSION,
} from '../actions/actions.js'

function smif(
    state = {
        isFetching: true,
        item: {}
    },
    action
) {
    switch (action.type){
    case REQUEST_SMIF_DETAILS:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_SMIF_DETAILS:
        return Object.assign({}, state, {
            isFetching: false,
            item: action.smif,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function model_runs(
    state = {
        isFetching: false,
        items: []
    },
    action
) {
    switch (action.type){
    case REQUEST_MODEL_RUNS:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_MODEL_RUNS:
        return Object.assign({}, state, {
            isFetching: false,
            items: action.model_runs,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function model_run(
    state = {
        isFetching: true,
        item: {}
    },
    action
) {
    switch (action.type){
    case REQUEST_MODEL_RUN:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_MODEL_RUN:
        return Object.assign({}, state, {
            isFetching: false,
            item: action.model_run,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function model_run_status(
    state = {
        isFetching: true,
        item: {}
    },
    action
) {
    switch (action.type){
    case REQUEST_MODEL_RUN_STATUS:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_MODEL_RUN_STATUS:
        return Object.assign({}, state, {
            isFetching: false,
            item: action.model_run_status,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function sos_models(
    state = {
        isFetching: true,
        items: [],
        error: {}
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
            items: action.data,
            error: action.error,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function sos_model(
    state = {
        isFetching: false,
        item: {},
        error: {}
    },
    action
) {
    switch (action.type){
    case REQUEST_SOS_MODEL:
        return Object.assign({}, state, {
            isFetching: true,
        })
    case RECEIVE_SOS_MODEL:
        return Object.assign({}, state, {
            isFetching: false,
            item: action.data,
            error: action.error
        })
    case SEND_SOS_MODEL:
        return Object.assign({}, state, {
            isFetching: true,
        })
    case REJECT_SOS_MODEL:
        return Object.assign({}, state, {
            isFetching: false,
            item: action.data,
            error: action.error
        })
    case ACCEPT_SOS_MODEL:
        return Object.assign({}, state, {
            isFetching: false,
            error: action.error
        })
    default:
        return state
    }
}

function sector_models(
    state = {
        isFetching: true,
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
        isFetching: true,
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

function scenarios(
    state = {
        isFetching: true,
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

function scenario(
    state = {
        isFetching: true,
        item: {}
    },
    action
) {
    switch (action.type){
    case REQUEST_SCENARIO:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_SCENARIO:
        return Object.assign({}, state, {
            isFetching: false,
            item: action.scenario,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function narratives(
    state = {
        isFetching: true,
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

function narrative(
    state = {
        isFetching: true,
        item: {}
    },
    action
) {
    switch (action.type){
    case REQUEST_NARRATIVE:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_NARRATIVE:
        return Object.assign({}, state, {
            isFetching: false,
            item: action.narrative,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function dimensions(
    state = {
        isFetching: true,
        items: []
    },
    action
) {
    switch (action.type){
    case REQUEST_DIMENSIONS:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_DIMENSIONS:
        return Object.assign({}, state, {
            isFetching: false,
            items: action.dimensions,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function dimension(
    state = {
        isFetching: true,
        item: {}
    },
    action
) {
    switch (action.type){
    case REQUEST_DIMENSION:
        return Object.assign({}, state, {
            isFetching: true
        })
    case RECEIVE_DIMENSION:
        return Object.assign({}, state, {
            isFetching: false,
            item: action.dimension,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

const rootReducer = combineReducers({
    smif,
    model_runs,
    model_run,
    model_run_status,
    sos_models,
    sos_model,
    sector_models,
    sector_model,
    scenarios,
    scenario,
    narratives,
    narrative,
    dimensions,
    dimension
})

export default rootReducer
