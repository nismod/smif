import React, { Component } from 'react'
import PropTypes from 'prop-types'

class NarrativeSelector extends Component {
    constructor(props) {
        super(props)

        this.handleChange = this.handleChange.bind(this)
    }

    pickSosModelByName(sos_model_name, sos_models) {
        /**
         * Get SosModel parameters, that belong to a given sos_model_name
         *
         * Arguments
         * ---------
         * sos_model_name: str
         *     Name identifier of the sos_model
         * sos_models: array
         *     Full list of available sos_models
         *
         * Returns
         * -------
         * Object
         *     All sos_model parameters that belong to the given sos_model_name
         */

        let sos_model = sos_models.filter(
            (sos_model) => sos_model.name === sos_model_name
        )[0]

        if (typeof sos_model === 'undefined') {
            sos_model = sos_models[0]
        }

        return sos_model
    }

    pickNarrativesBySets(narrative_sets, narratives) {
        /**
         * Get all the narratives, that belong to a given narrative_sets
         *
         * Arguments
         * ---------
         * narrative_sets: str
         *     Name identifier of the narrative_sets
         * narratives: array
         *     Full list of available narratives
         *
         * Returns
         * -------
         * Object
         *     All narratives that belong to the given narrative_sets
         */

        let narratives_in_sets = new Object()

        for (let i = 0; i < narrative_sets.length; i++) {

            // Get all narratives that belong to this narrative set
            narratives_in_sets[narrative_sets[i]] = narratives.filter(narratives => narratives.narrative_set === narrative_sets[i])

        }
        return narratives_in_sets
    }

    flagActiveNarratives(selectedNarratives, sosModelRun) {
        /**
         * Flag the narratives that are active in the project configuration
         *
         * Arguments
         * ---------
         *
         * Returns
         * -------
         * Object
         *     All narratives complimented with a true or false active flag
         */

        Object.keys(selectedNarratives).forEach(function(narrativeSet) {

            for (let i = 0; i < selectedNarratives[narrativeSet].length; i++) {

                selectedNarratives[narrativeSet][i].active = false
                
                if (typeof sosModelRun.narratives[selectedNarratives[narrativeSet][i]['narrative_set']] != 'undefined'){
                    if (sosModelRun.narratives[selectedNarratives[narrativeSet][i]['narrative_set']].includes(selectedNarratives[narrativeSet][i]['name'])) {
                        selectedNarratives[narrativeSet][i].active = true
                    }
                }
            }
        })

        return selectedNarratives
    }

    handleChange(event, narrative_name) {
        const target = event.target
        const {onChange} = this.props
        onChange(target.name, narrative_name, target.checked)
    }

    renderNarrativeSelector(selectedNarratives) {
        return (
            <div>
                {
                    Object.keys(selectedNarratives).map((narrativeSet) => (
                        <div key={narrativeSet}>
                            <div className="card" >
                                <div className="card-body">
                                    <h6 className="card-title">{narrativeSet}</h6>
                                    {
                                        selectedNarratives[narrativeSet].map((narrative) => (
                                            <div className="form-check" key={narrative.name}>
                                                <label className="form-check-label">
                                                    <input id={narrative.name} className="form-check-input" type="checkbox" name={narrativeSet} key={narrative.name} value={narrative.name} defaultChecked={narrative.active} onClick={(event) => {this.handleChange(event, narrative.name);}}></input>
                                                    {narrative.name}
                                                </label>
                                            </div>
                                        ))
                                    }
                                </div>
                            </div>
                            <br/>
                        </div>
                    ))
                }
            </div>
        )
    }

    renderDanger(message) {
        return (
            <div id="narrative_selector_alert-danger" className="alert alert-danger">
                {message}
            </div>
        )
    }

    renderWarning(message) {
        return (
            <div id="narrative_selector_alert-warning" className="alert alert-warning">
                {message}
            </div>
        )
    }

    renderInfo(message) {
        return (
            <div id="narrative_selector_alert-info" className="alert alert-info">
                {message}
            </div>
        )
    }

    render() {
        const {sosModelRun, sosModels, narratives} = this.props

        let selectedSosModel = null
        let selectedNarratives = null

        if (sosModelRun == null || sosModelRun == undefined || Object.keys(sosModelRun).length == 0) {
            return this.renderDanger('There is no SosModelRun configured')
        } else if (sosModels == null || sosModels == undefined || sosModels[0] == null) {
            return this.renderDanger('There are no SosModels configured')
        } else if (narratives == null || narratives == undefined || narratives[0] == null) {
            return this.renderDanger('There are no Narratives configured')
        } else if (sosModelRun.sos_model == "" || sosModelRun.sos_model == null || sosModelRun.sos_model == undefined) {
            return this.renderDanger('There is no SosModel configured in the SosModelRun')
        } else {
            selectedSosModel = this.pickSosModelByName(sosModelRun.sos_model, sosModels)
            if (selectedSosModel.narrative_sets == null || selectedSosModel.narrative_sets == undefined || selectedSosModel.narrative_sets[0] == undefined) {
                return this.renderInfo('There are no NarrativeSets configured in the SosModel')
            }

            selectedNarratives = this.pickNarrativesBySets(selectedSosModel.narrative_sets, narratives)
            selectedNarratives = this.flagActiveNarratives(selectedNarratives, sosModelRun)

            return this.renderNarrativeSelector(selectedNarratives)
        }
    }
}

NarrativeSelector.propTypes = {
    sosModelRun: PropTypes.object,
    sosModels: PropTypes.array,
    narratives: PropTypes.array,
    onChange: PropTypes.func
}

export default NarrativeSelector
