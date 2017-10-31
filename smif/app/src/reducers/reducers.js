import { combineReducers } from 'redux';
import {
    FETCH_PROJECTS,
    REQUEST_SOS_MODEL_RUNS,
    RECEIVE_SOS_MODEL_RUNS
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

const rootReducer = combineReducers({
    sos_model_runs,
    projects
});

export default rootReducer;
