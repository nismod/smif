import React from 'react'
import PropTypes from 'prop-types'

const ModelRunSummary = (props) => (
    <dl className="row">
        <dt className="col-sm-3">Name</dt>
        <dd className="col-sm-9">{props.ModelRun.name}</dd>
                
        <dt className="col-sm-3">Description</dt>
        <dd className="col-sm-9">{props.ModelRun.description}</dd>

        <dt className="col-sm-3">Created</dt>
        <dd className="col-sm-9">{props.ModelRun.stamp}</dd>

        <dt className="col-sm-3">Model</dt>
        <dd className="col-sm-9">
            <a href={'/configure/sos-models/' + props.ModelRun.sos_model}>
                {props.ModelRun.sos_model}
            </a>
        </dd>

        <dt className="col-sm-3">Scenarios</dt>
        <dd className="col-sm-9">
            {Object.keys(props.ModelRun.scenarios).map(scen => (
                <div key={'sum_scen_' + scen}>
                    <a href={'/configure/scenarios/' + scen}>
                        {scen}: {props.ModelRun.scenarios[scen]}
                    </a>
                </div>
            ))}
        </dd>

        <dt className="col-sm-3">Narratives</dt>
        <dd className="col-sm-9">
            {Object.keys(props.ModelRun.narratives).map(
                nar_set => <div key={'sum_nar_set_' + nar_set}>{nar_set}: 
                    <ul>
                        {props.ModelRun.narratives[nar_set].map(
                            nar => <li key={'sum_nar_set_' + nar_set + '_nar_' + nar}>{nar}</li>
                        )}
                    </ul>
                </div>
            )}
        </dd>

        <dt className="col-sm-3">Timesteps</dt>
        <dd className="col-sm-9">{props.ModelRun.timesteps.map(timestep => <div key={'timestep_' + timestep}>{timestep}</div>)}</dd>
    </dl>
)

ModelRunSummary.propTypes = {
    ModelRun: PropTypes.object.isRequired
}

export {
    ModelRunSummary
}