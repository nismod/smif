import { combineReducers } from 'redux'
import {
    APP_FORM_EDIT,
    APP_FORM_SAVE,
    APP_FORM_DISCARD,
    APP_FORM_KEEP_EDITING,
    APP_FORM_SAVE_DONE,
    APP_FORM_ACCEPT,
    APP_FORM_REJECT,
    APP_NAVIGATE,
    APP_NAVIGATE_DONE,
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

function app(
    state = {
        formEdit: false,
        formReqSave: false,
        formReqCancel: false,
        formSaving: false,
        formError: false,
        redirect: ''
    },
    action
) {
    switch (action.type){
    case APP_FORM_EDIT:
        return Object.assign({}, state, {
            formEdit: true,
        })
    case APP_FORM_SAVE:
        return Object.assign({}, state, {
            formReqSave: true,
            formReqCancel: false
        })
    case APP_FORM_DISCARD:
        return Object.assign({}, state, {
            formEdit: false,
            formReqSave: false,
            formSaving: false,
            formError: false,
            formReqCancel: false
        })
    case APP_FORM_SAVE_DONE:
        return Object.assign({}, state, {
            formReqSave: false,
            formSaving: true
        })
    case APP_FORM_KEEP_EDITING:
        return Object.assign({}, state, {
            formReqCancel: false,
        })
    case APP_FORM_ACCEPT:
        return Object.assign({}, state, {
            formEdit: false,
            formReqSave: false,
            formSaving: false,
            formError: false,
            formReqCancel: false
        })    
    case APP_FORM_REJECT:
        return Object.assign({}, state, {
            formReqSave: false,
            formSaving: false,
            formError: true,
            formReqCancel: false
        })
    case APP_NAVIGATE:
        return Object.assign({}, state, {
            redirect: action.data,
            formReqCancel: true
        })
    case APP_NAVIGATE_DONE:
        return Object.assign({}, state, {
            redirect: '',
            formReqCancel: false
        })
    
    default:
        return state
    }
}

function smif(
    state = {
        isFetching: true,
        item: {},
        error: {}
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
            item: action.data,
            error: action.error,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function model_runs(
    state = {
        isFetching: false,
        items: [],
        error: {}
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
            items: action.data,
            error: action.error,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function model_run(
    state = {
        isFetching: true,
        item: {},
        error: {}
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
            item: action.data,
            error: action.error,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function model_run_status(
    state = {
        isFetching: true,
        item: {},
        error: {}
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
            item: action.data,
            error: action.error,
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
        isFetching: true,
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
            error: action.error,
            lastUpdated: action.receivedAt
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
        items: [],
        error: {}
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
            items: action.data,
            error: action.error,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function sector_model(
    state = {
        isFetching: true,
        item: {},
        error: {}
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
            item: action.data,
            error: action.error,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function scenarios(
    state = {
        isFetching: true,
        items: [],
        error: {}
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
            items: action.data,
            error: action.error,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function scenario(
    state = {
        isFetching: true,
        item: {},
        error: {}
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
            item: action.data,
            error: action.error,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function narratives(
    state = {
        isFetching: true,
        items: [],
        error: {}
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
            items: action.data,
            error: action.error,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function narrative(
    state = {
        isFetching: true,
        item: {},
        error: {}
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
            item: action.data,
            error: action.error,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function dimensions(
    state = {
        isFetching: true,
        items: [],
        error: {}
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
            items: action.data,
            error: action.error,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

function dimension(
    state = {
        isFetching: true,
        item: {},
        error: {}
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
            item: action.data,
            error: action.error,
            lastUpdated: action.receivedAt
        })
    default:
        return state
    }
}

const rootReducer = combineReducers({
    app,
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
