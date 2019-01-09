import React, { Component } from 'react'
import PropTypes from 'prop-types'

import Ansi from 'ansi-to-react'
import moment from 'moment'
import stripAnsi from 'strip-ansi'
import { FaAngleDoubleUp, FaAngleDoubleDown, FaSave } from 'react-icons/fa'

class ConsoleDisplay extends Component {

    constructor(props) {
        super(props)
        this.anchor = React.createRef()

        this.state = {
            followConsole: false
        }
    }

    componentDidUpdate() {
        const {status} = this.props

        // Scroll the console output down during running status
        if (this.anchor.current != undefined && status == 'running' && this.state.followConsole) {
            this.anchor.current.scrollIntoView({ behavior: 'instant' })
        }
    }

    download(filename, text) {
        var element = document.createElement('a')
        element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text))
        element.setAttribute('download', filename)

        element.style.display = 'none'
        document.body.appendChild(element)

        element.click()
        document.body.removeChild(element)
    }

    render() {
        const { output, name } = this.props

        return (
            <div className="row">
                <div className="col-10">
                    {output.split(/\r?\n/).map((status_output, i) =>
                        <div key={'st_out_line_' + i}><Ansi>{status_output}</Ansi></div>
                    )}
                    <div className="cont" ref={this.anchor}/>
                </div>
                <div className={'col-2' + ((this.state.followConsole) ? ' align-self-end' : '')}>

                    <button
                        id="btn_download"
                        type="button"
                        className="btn btn-outline-dark btn-margin"
                        onClick={() => {
                            this.download(moment().format('YMMDD_HHmm') + '_' + name + '.log', stripAnsi(output))
                        }}>
                        <FaSave/>
                    </button>
                    <button
                        id="btn_toggle_scroll"
                        type="button"
                        className="btn btn-outline-dark btn-margin"
                        onClick={() => {
                            this.setState({followConsole: !this.state.followConsole})
                            if ( !this.state.followConsole) {
                                this.anchor.current.scrollIntoView({behavior: 'instant'})
                            } else {
                                window.scrollTo(0, 0)
                            }
                        }}>
                        {this.state.followConsole ? (
                            <FaAngleDoubleUp/>
                        ) : (
                            <FaAngleDoubleDown/>
                        )}
                    </button>
                </div>
            </div>
        )
    }
}

ConsoleDisplay.propTypes= {
    name: PropTypes.string,
    output: PropTypes.string,
    status: PropTypes.string
}

export default ConsoleDisplay
