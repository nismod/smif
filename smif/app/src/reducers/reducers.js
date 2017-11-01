import { combineReducers } from 'redux';
import {
    FETCH_PROJECTS,
    REQUEST_SOS_MODEL_RUNS,
    RECEIVE_SOS_MODEL_RUNS,
    REQUEST_SOS_MODEL_RUN,
    RECEIVE_SOS_MODEL_RUN
} from '../actions/actions.js';

function projects(state = {}, action) {
    switch (action.type) {
    case FETCH_PROJECTS:
        return ['Project1', 'Project2', 'Project3'];
    default:
        return state;
    }
};

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
        });
    case RECEIVE_SOS_MODEL_RUNS:
        return Object.assign({}, state, {
            isFetching: false,
            items: action.sos_model_runs,
            lastUpdated: action.receivedAt
        });
    default:
        return state;
    }
}

function sos_model_run(
    state = {
        isFetching: false,
        item: []
    },
    action
) {
    switch (action.type){
    case REQUEST_SOS_MODEL_RUN:
        return Object.assign({}, state, {
            isFetching: true
        });
    case RECEIVE_SOS_MODEL_RUN:
        return Object.assign({}, state, {
            isFetching: false,
            item: action.sos_model_run,
            lastUpdated: action.receivedAt
        });
    default:
        return state;
    }
}

const rootReducer = combineReducers({
    sos_model_runs,
    sos_model_run,
    projects
});

export default rootReducer;
