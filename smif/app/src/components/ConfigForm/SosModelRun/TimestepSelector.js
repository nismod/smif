import React, { Component } from 'react';
import PropTypes from 'prop-types'

class TimestepSelector extends Component {
    constructor(props) {
        super(props)

        this.onChangeHandler = this.onChangeHandler.bind(this)

        this.state = {}
        this.state.baseYear = null
        this.state.endYear = null
        this.state.resolution = null
    }
    
    componentWillMount(){
        const {timeSteps} = this.props

        if (timeSteps != null ){
            this.setState({baseYear: timeSteps[0]})
            this.setState({endYear: timeSteps[timeSteps.length - 1]})
            this.setState({resolution: ((timeSteps[timeSteps.length - 1] - timeSteps[0]) / (timeSteps.length - 1))})
        }
    }

    createBaseyearSelectItems() {
        let years = []
            
        for (let i = 2000; i <= 2100; i++) {
            years.push(<option key={'baseyear_' + i} value={i}>{i}</option>)
        }

        return years
    }

    createEndyearSelectItems(baseYear, resolution) {
        let years = []

        for (let i = baseYear; i <= baseYear + 100; i = i + resolution) {
            years.push(<option key={'endyear_' + i} value={i}>{i}</option>)
        }
        return years
    }

    onChangeHandler(event) {
        const {onChange} = this.props
        let {baseYear, endYear, resolution} = this.state

        const target = event.target
        const value = target.type === 'checkbox' ? target.checked : target.value
        const name = target.name

        if (name == 'baseyear') {
            baseYear = parseInt(value)
            this.setState({baseYear: parseInt(value)})
            this.setState({endYear: parseInt(value)})
        } else if (name == 'endyear') {
            endYear = parseInt(value)
            this.setState({endYear: parseInt(value)})
        } else if (name == 'resolution') {
            resolution = parseInt(value)
            this.setState({resolution: parseInt(value)})
            this.setState({endYear: baseYear})
        }

        let timesteps = []
        if (endYear > baseYear) {
            // Calculate new timesteps
            for (let i = baseYear; i <= endYear; i+=resolution)
            {
                timesteps.push(i)
            }
        } else {
            timesteps = [baseYear]
        }

        onChange(timesteps)
    }

    renderTimestepSelector(baseYear, endYear, resolution) {
        return (
            <div>

                <div className="form-group row">
                    <label className="col-sm-2 col-form-label">Resolution</label>
                    <div className="col-sm-10">
                        <input className="form-control" name="resolution" type="number" min="1" defaultValue={resolution} onChange={this.onChangeHandler}/>
                    </div>
                </div>

                <div className="form-group row">
                    <label className="col-sm-2 col-form-label">Base year</label>
                    <div className="col-sm-10">
                        <select className="form-control" value={baseYear} name="baseyear" onChange={this.onChangeHandler}>
                            <option disabled="disabled">Please select a base year</option>
                            {this.createBaseyearSelectItems()}
                        </select>
                    </div>
                </div>

                <div className="form-group row">
                    <label className="col-sm-2 col-form-label">End year</label>
                    <div className="col-sm-10">
                        <select className="form-control" value={endYear} name="endyear" onChange={this.onChangeHandler}>
                            <option value="" disabled="disabled">Please select an end year</option>
                            {this.createEndyearSelectItems(baseYear, resolution)}
                        </select>
                    </div>
                </div>
            </div>
        )
    }

    renderWarning(message) {
        return (
            <div className="alert alert-danger">
                {message}
            </div>
        )
    }

    render() {
        const {timeSteps} = this.props
        const {baseYear, endYear, resolution} = this.state

        if (timeSteps == null || timeSteps == undefined || Object.keys(timeSteps).length === 0) {
            return this.renderWarning('There is no defaultValue configured')
        } else if (baseYear == null) {
            return this.renderWarning('Loading defaultValues')
        } else {
            return this.renderTimestepSelector(baseYear, endYear, resolution)
        }
    }
}

TimestepSelector.propTypes = {
    timeSteps: PropTypes.array,
    onChange: PropTypes.func
}

export default TimestepSelector