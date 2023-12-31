import React, { Component } from 'react';
import { Button, Form, Radio } from 'semantic-ui-react';
import { isMobile } from 'react-device-detect';
import { connect } from 'react-redux';
import { createSelector } from 'reselect';
import {
  checkedRegionsLabels,
  mapViewsWithNoFlags,
} from '../../assets/mapViewSettings';
import { capitalize } from '../../helpers/textHelpers';
import QuizMenu from '../styles/QuizMenuStyles';
import { setRegionCheckbox, tooltipToggle } from '../../actions/mapActions';
import {
  changeQuiz,
  startQuiz,
  closeQuiz,
  setLabel,
  toggleExternalRegions,
  toggleTimer,
} from '../../actions/quizActions';

// const generateQuizOptions = regionType => [
//   { label: `Click ${regionType}`, value: 'click_name_ordered' },
//   { label: `Type Marked ${regionType}`, value: 'type_name_ordered' },
//   { label: `Type any ${regionType}`, value: 'type_name_unordered' },
//   { label: 'Click Capital', value: 'click_capital_ordered' },
//   { label: 'Type Marked Capital', value: 'type_capital_ordered' },
//   { label: 'Type any Capital', value: 'type_capital_unordered' },
//   { label: `Click ${regionType} from Flag`, value: 'click_flag_ordered' },
// ];

class QuizBox extends Component {
  constructor() {
    super();

    this.state = { regionMenu: false };
  }

  handleQuizChange = (event, { value }) => {
    this.props.changeQuiz(value);
  };

  handleLabelToggle = (event, data) => {
    const marker = data.value;
    const { setLabel, quiz } = this.props;
    const { markerToggle } = quiz;
    const parentMarker =
      markerToggle === '' || marker !== markerToggle ? marker : '';
    setLabel(parentMarker);
  };

  handleRegionMenu = () => {
    this.setState({ regionMenu: !this.state.regionMenu });
  };

  handleCheckBox = event => {
    const regionName = event.target.value;
    const { setRegionCheckbox } = this.props;
    setRegionCheckbox(regionName);
  };

  // handleQuizOptions = subRegionNameCap => {
  //   const { quizType } = this.props.quiz;
  //   const { currentMap } = this.props.map;
  //   let quizOptions = generateQuizOptions(subRegionNameCap);
  //   if (mapViewsWithNoFlags.includes(currentMap)) {
  //     const idx = quizOptions.findIndex(obj => obj.value === 'click_flag');
  //     quizOptions.splice(idx, 1);
  //   }
  //   return quizOptions.map(form => (
  //     <Form.Field key={form.value}>
  //       <Radio
  //         aria-label={form.label}
  //         label={form.label}
  //         value={form.value}
  //         name="quiz"
  //         checked={quizType === form.value}
  //         onChange={this.handleQuizChange}
  //       />
  //     </Form.Field>
  //   ));
  // };

  start = () => {
    this.props.startQuiz();
  };

  render() {
    const { regionMenu } = this.state;
    const {
      quiz,
      map,
      toggleExternalRegions,
      tooltipToggle,
      toggleTimer,
    } = this.props;
    const { markerToggle, areExternalRegionsOnQuiz, isTimerEnabled } = quiz;
    const { checkedRegions, currentMap, subRegionName, tooltip } = map;
    const regionLabel = markerToggle === 'name';
    const capitalLabel = markerToggle === 'capital';
    const formSize = isMobile ? 'mini' : 'small';
    const subRegionNameCap = capitalize(subRegionName);

    return (
      <QuizMenu regionMenu={regionMenu}>
        <div>
          <Button
            size={formSize}
            onClick={this.start}
            className="startButton"
            aria-label="start quiz"
          >
            Hacer Predicción
          </Button>
          
          {currentMap !== 'World' && (
            <div className="App-quiz-toggle">
              {checkedRegionsLabels.includes(currentMap) && (
                <Button
                  toggle
                  size={formSize}
                  active={areExternalRegionsOnQuiz}
                  onClick={toggleExternalRegions}
                  aria-label="toggle external regions for quizzes"
                  style={{ width: '9em', margin: '1.5em 0', padding: '0.8em' }}
                >
                  {'Include external regions'}
                </Button>
              )}
              <div className="App-quiz-toggle-header">TOGGLE LABEL</div>
              <Button.Group size={formSize} compact>
                <Button
                  toggle
                  active={regionLabel}
                  value="name"
                  onClick={this.handleLabelToggle}
                  aria-label="toggle region names"
                >
                  {subRegionNameCap}
                </Button>
                <Button.Or aria-label="or" />
                <Button
                  toggle
                  active={capitalLabel}
                  value="capital"
                  onClick={this.handleLabelToggle}
                  aria-label="toggle region capitals"
                >
                  {'Capital'}
                </Button>
              </Button.Group>
            </div>
          )}
          {currentMap === 'World' && (
            <Form className="fmRegionSelect">
              {checkedRegionsLabels.map(region => (
                <Form.Field
                  aria-label={region}
                  label={region}
                  value={region}
                  key={region}
                  control="input"
                  type="checkbox"
                  checked={checkedRegions.includes(region)}
                  onChange={this.handleCheckBox}
                />
              ))}
            </Form>
          )}

          <div style={{ marginTop: '2rem' }}>
            <Radio
              slider
              fitted
              size={formSize}
              label={`Tooltip`}
              checked={tooltip}
              onChange={tooltipToggle}
              style={{}}
            />
          </div>

          <div style={{ marginTop: '2rem' }}>
            <Radio
              slider
              fitted
              size={formSize}
              label={`Timer`}
              checked={isTimerEnabled}
              onChange={toggleTimer}
              style={{}}
            />
          </div>
        </div>
      </QuizMenu>
    );
  }
}

const getAppState = createSelector(
  state => state.map.checkedRegions,
  state => state.map.currentMap,
  state => state.map.subRegionName,
  state => state.map.tooltip,
  state => state.quiz.quizType,
  state => state.quiz.markerToggle,
  state => state.quiz.areExternalRegionsOnQuiz,
  state => state.quiz.isTimerEnabled,
  (
    checkedRegions,
    currentMap,
    subRegionName,
    tooltip,
    quizType,
    markerToggle,
    areExternalRegionsOnQuiz,
    isTimerEnabled
  ) => ({
    map: { checkedRegions, currentMap, subRegionName, tooltip },
    quiz: { quizType, markerToggle, areExternalRegionsOnQuiz, isTimerEnabled },
  })
);

export default connect(getAppState, {
  setRegionCheckbox,
  changeQuiz,
  startQuiz,
  closeQuiz,
  setLabel,
  tooltipToggle,
  toggleExternalRegions,
  toggleTimer,
})(QuizBox);
