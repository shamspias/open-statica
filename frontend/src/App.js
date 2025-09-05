import React, {useState} from 'react';
import {Layout, Menu, Upload, Button, Table, Card, Select, message, Tabs, Row, Col, Statistic, Space, Spin} from 'antd';
import {UploadOutlined, BarChartOutlined, LineChartOutlined, DotChartOutlined, TableOutlined} from '@ant-design/icons';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';

const {Header, Sider, Content} = Layout;
const {Option} = Select;

const API_URL = 'http://localhost:8000/api';

function App() {
    const [sessionId, setSessionId] = useState(null);
    const [data, setData] = useState(null);
    const [columns, setColumns] = useState([]);
    const [numericColumns, setNumericColumns] = useState([]);
    const [categoricalColumns, setCategoricalColumns] = useState([]);
    const [selectedMenu, setSelectedMenu] = useState('upload');
    const [analysisResults, setAnalysisResults] = useState(null);
    const [chart, setChart] = useState(null);
    const [loading, setLoading] = useState(false);
    const [selectedX, setSelectedX] = useState(null);
    const [selectedY, setSelectedY] = useState(null);

    const menuItems = [
        {
            key: 'upload',
            icon: <UploadOutlined/>,
            label: 'Upload Data',
        },
        {
            key: 'data',
            icon: <TableOutlined/>,
            label: 'View Data',
            disabled: !sessionId,
        },
        {
            key: 'descriptive',
            icon: <BarChartOutlined/>,
            label: 'Descriptive Stats',
            disabled: !sessionId,
        },
        {
            key: 'visualization',
            icon: <LineChartOutlined/>,
            label: 'Visualizations',
            disabled: !sessionId,
        },
        {
            key: 'tests',
            icon: <DotChartOutlined/>,
            label: 'Statistical Tests',
            disabled: !sessionId,
        },
    ];

    const handleFileUpload = async (file) => {
        const formData = new FormData();
        formData.append('file', file);

        setLoading(true);
        try {
            const response = await axios.post(`${API_URL}/upload`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setSessionId(response.data.session_id);
            setData(response.data.preview);
            setColumns(response.data.column_names);
            setNumericColumns(response.data.numeric_columns);
            setCategoricalColumns(response.data.categorical_columns);
            message.success('File uploaded successfully!');
            setSelectedMenu('data');
        } catch (error) {
            console.error('Upload error:', error);
            message.error('Failed to upload file');
        } finally {
            setLoading(false);
        }
        return false;
    };

    const runDescriptiveStats = async () => {
        if (!sessionId) return;

        setLoading(true);
        try {
            const response = await axios.post(`${API_URL}/statistics/descriptive`, {
                session_id: sessionId,
                columns: numericColumns
            });
            setAnalysisResults(response.data);
            message.success('Analysis completed!');
        } catch (error) {
            console.error('Analysis error:', error);
            message.error('Analysis failed');
        } finally {
            setLoading(false);
        }
    };

    const runCorrelation = async () => {
        if (!sessionId) return;

        setLoading(true);
        try {
            const response = await axios.post(`${API_URL}/statistics/correlation`, {
                session_id: sessionId,
                columns: numericColumns
            });

            const chartResponse = await axios.post(`${API_URL}/visualization/create`, {
                session_id: sessionId,
                type: 'correlation_heatmap',
                columns: numericColumns
            });

            setAnalysisResults(response.data);
            setChart(chartResponse.data.chart);
            message.success('Correlation analysis completed!');
        } catch (error) {
            console.error('Correlation error:', error);
            message.error('Analysis failed');
        } finally {
            setLoading(false);
        }
    };

    const createVisualization = async (type, params = {}) => {
        if (!sessionId) return;

        setLoading(true);
        try {
            const response = await axios.post(`${API_URL}/visualization/create`, {
                session_id: sessionId,
                type,
                ...params
            });
            setChart(response.data.chart);
            message.success('Chart created!');
        } catch (error) {
            console.error('Visualization error:', error);
            message.error('Failed to create chart');
        } finally {
            setLoading(false);
        }
    };

    const createScatterPlot = () => {
        if (selectedX && selectedY) {
            createVisualization('scatter', {x: selectedX, y: selectedY});
        } else {
            message.warning('Please select both X and Y axes');
        }
    };

    const renderDataTable = () => {
        if (!data || data.length === 0) return <p>No data available</p>;

        const tableColumns = columns.map(col => ({
            title: col,
            dataIndex: col,
            key: col,
            render: (text) => text !== null && text !== undefined ? String(text) : 'N/A',
            ellipsis: true,
        }));

        return (
            <Table
                columns={tableColumns}
                dataSource={data.map((row, index) => ({...row, key: index}))}
                scroll={{x: 'max-content'}}
                pagination={{pageSize: 10}}
            />
        );
    };

    const renderDescriptiveStats = () => {
        if (!analysisResults) return null;

        return (
            <div>
                {Object.entries(analysisResults).map(([column, stats]) => (
                    <Card key={column} title={column} style={{marginBottom: 16}}>
                        {typeof stats === 'object' && stats.count !== undefined ? (
                            <Row gutter={16}>
                                <Col span={6}>
                                    <Statistic
                                        title="Mean"
                                        value={stats.mean ? stats.mean.toFixed(2) : 'N/A'}
                                    />
                                </Col>
                                <Col span={6}>
                                    <Statistic
                                        title="Median"
                                        value={stats.median ? stats.median.toFixed(2) : 'N/A'}
                                    />
                                </Col>
                                <Col span={6}>
                                    <Statistic
                                        title="Std Dev"
                                        value={stats.std ? stats.std.toFixed(2) : 'N/A'}
                                    />
                                </Col>
                                <Col span={6}>
                                    <Statistic
                                        title="Count"
                                        value={stats.count || 0}
                                    />
                                </Col>
                            </Row>
                        ) : (
                            <p>Categorical variable - Mode: {stats.mode}</p>
                        )}
                    </Card>
                ))}
            </div>
        );
    };

    return (
        <Layout style={{height: '100vh'}}>
            <Header style={{background: '#fff', padding: '0 20px', borderBottom: '1px solid #f0f0f0'}}>
                <h1 style={{margin: 0, color: '#1890ff'}}>OpenStatica - Statistical Analysis Platform</h1>
            </Header>

            <Layout>
                <Sider width={200} style={{background: '#fff'}}>
                    <Menu
                        mode="inline"
                        selectedKeys={[selectedMenu]}
                        onClick={(e) => setSelectedMenu(e.key)}
                        items={menuItems}
                        style={{height: '100%', borderRight: 0}}
                    />
                </Sider>

                <Content style={{padding: 24, background: '#f0f2f5', overflow: 'auto'}}>
                    <Spin spinning={loading}>
                        {selectedMenu === 'upload' && (
                            <Card title="Upload Data File">
                                <Upload
                                    accept=".csv,.xlsx,.xls"
                                    beforeUpload={handleFileUpload}
                                    showUploadList={false}
                                >
                                    <Button icon={<UploadOutlined/>} type="primary" size="large">
                                        Select File (CSV or Excel)
                                    </Button>
                                </Upload>
                                <p style={{marginTop: 16}}>
                                    Upload your data file to start analysis. Supported formats: CSV, Excel
                                </p>
                            </Card>
                        )}

                        {selectedMenu === 'data' && sessionId && (
                            <Card title="Data Preview">
                                {renderDataTable()}
                            </Card>
                        )}

                        {selectedMenu === 'descriptive' && sessionId && (
                            <Card title="Descriptive Statistics">
                                <Space style={{marginBottom: 16}}>
                                    <Button type="primary" onClick={runDescriptiveStats}>
                                        Run Analysis
                                    </Button>
                                    <Button onClick={runCorrelation}>
                                        Correlation Matrix
                                    </Button>
                                </Space>
                                {renderDescriptiveStats()}
                                {chart && (
                                    <div style={{marginTop: 20}}>
                                        <Plot {...chart} layout={{...chart.layout, autosize: true}}
                                              style={{width: '100%'}}/>
                                    </div>
                                )}
                            </Card>
                        )}

                        {selectedMenu === 'visualization' && sessionId && (
                            <Card title="Data Visualization">
                                <Tabs
                                    defaultActiveKey="histogram"
                                    items={[
                                        {
                                            key: 'histogram',
                                            label: 'Histogram',
                                            children: (
                                                <>
                                                    <Select
                                                        style={{width: 200, marginBottom: 16}}
                                                        placeholder="Select column"
                                                        onChange={(value) => createVisualization('histogram', {column: value})}
                                                    >
                                                        {numericColumns.map(col => (
                                                            <Option key={col} value={col}>{col}</Option>
                                                        ))}
                                                    </Select>
                                                </>
                                            ),
                                        },
                                        {
                                            key: 'scatter',
                                            label: 'Scatter Plot',
                                            children: (
                                                <>
                                                    <Space style={{marginBottom: 16}}>
                                                        <Select
                                                            style={{width: 200}}
                                                            placeholder="X axis"
                                                            onChange={(value) => setSelectedX(value)}
                                                        >
                                                            {numericColumns.map(col => (
                                                                <Option key={col} value={col}>{col}</Option>
                                                            ))}
                                                        </Select>
                                                        <Select
                                                            style={{width: 200}}
                                                            placeholder="Y axis"
                                                            onChange={(value) => setSelectedY(value)}
                                                        >
                                                            {numericColumns.map(col => (
                                                                <Option key={col} value={col}>{col}</Option>
                                                            ))}
                                                        </Select>
                                                        <Button type="primary" onClick={createScatterPlot}>
                                                            Create Plot
                                                        </Button>
                                                    </Space>
                                                </>
                                            ),
                                        },
                                        {
                                            key: 'box',
                                            label: 'Box Plot',
                                            children: (
                                                <Button
                                                    type="primary"
                                                    onClick={() => createVisualization('box', {columns: numericColumns})}
                                                >
                                                    Create Box Plot
                                                </Button>
                                            ),
                                        },
                                    ]}
                                />

                                {chart && (
                                    <div style={{marginTop: 20}}>
                                        <Plot
                                            {...chart}
                                            layout={{...chart.layout, autosize: true}}
                                            style={{width: '100%'}}
                                            useResizeHandler={true}
                                        />
                                    </div>
                                )}
                            </Card>
                        )}

                        {selectedMenu === 'tests' && sessionId && (
                            <Card title="Statistical Tests">
                                <Tabs
                                    defaultActiveKey="ttest"
                                    items={[
                                        {
                                            key: 'ttest',
                                            label: 'T-Test',
                                            children: <p>Configure and run t-tests (Coming soon)</p>,
                                        },
                                        {
                                            key: 'anova',
                                            label: 'ANOVA',
                                            children: <p>Configure and run ANOVA (Coming soon)</p>,
                                        },
                                        {
                                            key: 'chi',
                                            label: 'Chi-Square',
                                            children: <p>Configure and run chi-square test (Coming soon)</p>,
                                        },
                                        {
                                            key: 'regression',
                                            label: 'Regression',
                                            children: <p>Configure and run regression analysis (Coming soon)</p>,
                                        },
                                    ]}
                                />
                            </Card>
                        )}
                    </Spin>
                </Content>
            </Layout>
        </Layout>
    );
}

export default App;