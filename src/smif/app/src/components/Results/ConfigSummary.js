import React from 'react'

const SosModelRunSummary = (props) => (
    <dl className="row">
        <dt className="col-sm-3">Name</dt>
        <dd className="col-sm-9">{props.sosModelRun.name}</dd>
                
        <dt className="col-sm-3">Description</dt>
        <dd className="col-sm-9">{props.sosModelRun.description}</dd>

        <dt className="col-sm-3">Created</dt>
        <dd className="col-sm-9">{props.sosModelRun.stamp}</dd>

        <dt className="col-sm-3">Model</dt>
        <dd className="col-sm-9">{props.sosModelRun.sos_model}</dd>

        <dt className="col-sm-3">Scenarios</dt>
        <dd className="col-sm-9">
            {Object.keys(props.sosModelRun.scenarios).map(
                scen => <div key={'sum_scen_' + scen}>{scen}: {props.sosModelRun.scenarios[scen]}</div>
            )}
        </dd>

        <dt className="col-sm-3">Narratives</dt>
        <dd className="col-sm-9">
            {Object.keys(props.sosModelRun.narratives).map(
                nar_set => <div key={'sum_nar_set_' + nar_set}>{nar_set}: 
                    <ul>
                        {props.sosModelRun.narratives[nar_set].map(
                            nar => <li key={'sum_nar_set_' + nar_set + '_nar_' + nar}>{nar}</li>
                        )}
                    </ul>
                </div>
            )}
        </dd>

        <dt className="col-sm-3">Timesteps</dt>
        <dd className="col-sm-9">{props.sosModelRun.timesteps.map(timestep => <div key={'timestep_' + timestep}>{timestep}</div>)}</dd>
    </dl>
)

module.exports = {
    SosModelRunSummary
}