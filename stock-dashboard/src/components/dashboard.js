import { SearchOutlined } from "@ant-design/icons";
import React, { useEffect, useState } from "react";
import { Button, Divider, Flex, Select, Spin, Table } from "antd";

const { Option } = Select;

const Dashboard = () => {
  const [selectedStock, setSelectedStock] = useState(undefined);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState("LightGBM");
  const [selectedParameter, setSelectedParameter] = useState("default");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [chartImage, setChartImage] = useState(null);

  const stockOptions = [
    "YEG",
    "VCF",
    "PTL",
    "KBC",
    "CCL",
    "VCB",
    "PVD",
    "CCI",
    "JVC",
    "ITD",
    "CDC",
    "VCA",
    "QCG",
    "ITC",
    "CHP",
    "PVT",
    "VAF",
    "KDC",
    "VCG",
    "VDS",
    "POW",
    "KOS",
    "BWE",
    "VDP",
    "PPC",
    "PTC",
    "KMR",
    "KHP",
    "C32",
    "VCI",
    "PTB",
    "KDH",
    "C47",
    "PSH",
    "RAL",
    "ITA",
    "CIG",
    "CLL",
    "ADG",
    "TV2",
    "SAV",
    "HVX",
    "SBA",
    "ICT",
    "HVN",
    "TTF",
    "SBT",
    "HVH",
    "CMG",
    "TTE",
    "SBV",
    "CLW",
    "VSC",
    "IDI",
    "SAB",
    "UIC",
    "RDP",
    "IMP",
    "CII",
    "TYA",
    "REE",
    "ILB",
    "ABS",
    "VTO",
    "CKG",
    "TVT",
    "S4A",
    "IJC",
    "CLC",
    "TVS",
    "BVH",
    "HUB",
    "KPF",
    "VGC",
    "BFC",
    "VNE",
    "PDR",
    "LSS",
    "BHN",
    "VMD",
    "MBB",
    "PET",
    "BIC",
    "VJC",
    "PGC",
    "LM8",
    "PGD",
    "LIX",
    "LPB",
    "VRE",
    "PDN",
    "BCM",
    "AST",
    "VOS",
    "OGC",
    "MIG",
    "BCE",
    "VNS",
    "VNG",
    "OPC",
    "BCG",
    "VNINDEX",
    "PAC",
    "MDG",
    "PC1",
    "MCP",
    "MHC",
    "BID",
    "VIX",
    "PGI",
    "VIB",
    "PLP",
    "LCG",
    "BRC",
    "VHM",
    "PLX",
    "BMP",
    "LBM",
    "L10",
    "BTP",
    "VHC",
    "PNC",
    "KSB",
    "BTT",
    "PMG",
    "LDG",
    "PJT",
    "VIC",
    "LHG",
    "ADS",
    "PHC",
    "LGL",
    "BKG",
    "VIP",
    "AAT",
    "YBM",
    "PHR",
    "LGC",
    "BMC",
    "VID",
    "PIT",
    "LEC",
    "BMI",
    "PNJ",
    "MSB",
    "CMV",
    "SC5",
    "STG",
    "HAR",
    "DBT",
    "THG",
    "STK",
    "HAH",
    "HAS",
    "DC4",
    "SVC",
    "HAG",
    "DCL",
    "TDW",
    "SVD",
    "GVR",
    "TEG",
    "DCM",
    "STB",
    "DBD",
    "SRF",
    "HCM",
    "VSH",
    "ACL",
    "SSB",
    "HCD",
    "TIP",
    "DAT",
    "SSI",
    "HBC",
    "DBC",
    "TIX",
    "ST8",
    "HAX",
    "TLD",
    "TDP",
    "SVI",
    "GTA",
    "GEX",
    "DHG",
    "TCT",
    "TCD",
    "GEG",
    "DHM",
    "TCB",
    "TCR",
    "FUCVREIT",
    "DIG",
    "TCO",
    "TCL",
    "DMC",
    "DLG",
    "TCH",
    "GIL",
    "TBC",
    "TDC",
    "SVT",
    "GSP",
    "DGC",
    "TDM",
    "SZC",
    "GMD",
    "DGW",
    "TDH",
    "SZL",
    "GMC",
    "ACC",
    "DHA",
    "TDG",
    "VSI",
    "DHC",
    "HDB",
    "TTA",
    "SRC",
    "DAH",
    "HT1",
    "CRE",
    "TPB",
    "SGN",
    "HSL",
    "SGR",
    "SFI",
    "HSG",
    "TNT",
    "SGT",
    "HRC",
    "CSV",
    "TNI",
    "SHA",
    "CSM",
    "HQC",
    "TPC",
    "HTI",
    "HU1",
    "CMX",
    "TSC",
    "SCR",
    "HTV",
    "SCS",
    "CRC",
    "HTN",
    "TRC",
    "SFC",
    "HTL",
    "COM",
    "TRA",
    "SFG",
    "CNG",
    "CTD",
    "TNH",
    "SHI",
    "SMA",
    "HHS",
    "CVT",
    "TMS",
    "SMB",
    "HHP",
    "HID",
    "D2D",
    "SMC",
    "HDG",
    "DAG",
    "TLH",
    "SPM",
    "HDC",
    "TMP",
    "SKG",
    "TMT",
    "CTS",
    "HPG",
    "CTF",
    "TNC",
    "SHP",
    "HNG",
    "SJD",
    "HMC",
    "CTG",
    "TNA",
    "ACB",
    "VTB",
    "CTI",
    "TN1",
    "SJS",
    "HII",
    "TLG",
    "OCB",
    "SAM",
    "TCM",
    "MSN",
    "NVL",
    "VPD",
    "NVT",
    "VPI",
    "ANV",
    "NT2",
    "NAF",
    "AGR",
    "NHA",
    "VPS",
    "NLG",
    "VPB",
    "NKG",
    "NCT",
    "MWG",
    "NTL",
    "ASP",
    "APH",
    "VPG",
    "NBB",
    "NHH",
    "ASM",
    "NNC",
    "MSH",
    "VPH",
    "AGG",
    "NAV",
    "APG",
    "VRC",
  ];
  const columns = [
    {
      title: "Chỉ số đánh giá trên tập train",
      dataIndex: "name",
    },

    {
      title: "Kết quả trên tập train",
      dataIndex: "address",
    },
  ];
  const columns1 = [
    {
      title: "Chỉ số đánh giá trên tập test",
      dataIndex: "name",
    },

    {
      title: "Kết quả trên tập test",
      dataIndex: "address",
    },
  ];
  const columns2 = [
    {
      title: "Ngày",
      dataIndex: "name",
    },

    {
      title: "Giá thực tế ngày trước",
      dataIndex: "address",
    },
    {
      title: "Giá dự đoán cuối cùng",
      dataIndex: "age",
    },
  ];
  const columns3 = [
    {
      title: "Chỉ số đánh giá trên tập train",
      dataIndex: "name",
    },

    {
      title: "Kết quả trên tập train",
      dataIndex: "address",
    },
  ];
  const data = [
    {
      key: "1",
      name: "Độ phù hợp tập train",
      age: 32,
      address: results[0]?.DoPhuHopTrain_XGB,
    },
    {
      key: "2",
      name: "Sai số tuyệt đối trung bình trên tập train (VNĐ)",
      age: 42,
      address: results[0]?.SaiSoTrungBinhTrain_XGB,
    },
    {
      key: "3",
      name: " Phần trăm sai số tuyệt đối trung bình tập train",
      age: 32,
      address: results[0]?.PhanTramSaiSoTrain_XGB,
    },
    {
      key: "4",
      name: " Root Mean Squared Error trên tập train (VNĐ)",
      age: 32,
      address: results[0]?.RMSETrain_XGB,
    },
    {
      key: "5",
      name: "Mean Squared Error trên tập train (VNĐ^2)",
      age: 32,
      address: results[0]?.MSETrain_XGB,
    },
  ];
  const data1 = [
    {
      key: "1",
      name: "Độ phù hợp tập test",
      age: 32,
      address: results[0]?.DoPhuHopTest_XGB,
    },
    {
      key: "2",
      name: "Sai số tuyệt đối trung bình trên tập test (VNĐ)",
      age: 42,
      address: results[0]?.SaiSoTrungBinhTest_XGB,
    },
    {
      key: "3",
      name: " Phần trăm sai số tuyệt đối trung bình tập test",
      age: 32,
      address: results[0]?.PhanTramSaiSoTest_XGB,
    },
    {
      key: "4",
      name: " Root Mean Squared Error trên tập test (VNĐ)",
      age: 32,
      address: results[0]?.RMSETest_XGB,
    },
    {
      key: "5",
      name: "Mean Squared Error trên tập test (VNĐ^2)",
      age: 32,
      address: results[0]?.MSETest_XGB,
    },
  ];
  const data2 = [
    {
      key: "1",
      name: results[0]?.GiaThucTeCuoi_XGB?.Ngay[0],
      address: results[0]?.GiaThucTeCuoi_XGB?.GiaNgayTruoc[0],
      age: results[0]?.GiaThucTeCuoi_XGB?.GiaDuDoan[0]
      
    },
    
  ];
  const data3 = [
    {
      key: "1",
      name: "Độ phù hợp tập train",
      age: 32,
      address: results[0]?.DoPhuHopTrain_LightGBM,
    },
    {
      key: "2",
      name: "Sai số tuyệt đối trung bình trên tập train (VNĐ)",
      age: 42,
      address: results[0]?.SaiSoTrungBinhTrain_LightGBM,
    },
    {
      key: "3",
      name: " Phần trăm sai số tuyệt đối trung bình tập train",
      age: 32,
      address: results[0]?.PhanTramSaiSoTrain_LightGBM,
    },
    {
      key: "4",
      name: " Root Mean Squared Error trên tập train (VNĐ)",
      age: 32,
      address: results[0]?.RMSETrain_LightGBM,
    },
    {
      key: "5",
      name: "Mean Squared Error trên tập train (VNĐ^2)",
      age: 32,
      address: results[0]?.MSETrain_LightGBM,
    },
  ];
  const data4 = [
    {
      key: "1",
      name: "Độ phù hợp tập test",
      age: 32,
      address: results[0]?.DoPhuHopTest_LightGBM,
    },
    {
      key: "2",
      name: "Sai số tuyệt đối trung bình trên tập test (VNĐ)",
      age: 42,
      address: results[0]?.SaiSoTrungBinhTest_LightGBM,
    },
    {
      key: "3",
      name: " Phần trăm sai số tuyệt đối trung bình tập test",
      age: 32,
      address: results[0]?.PhanTramSaiSoTest_LightGBM,
    },
    {
      key: "4",
      name: " Root Mean Squared Error trên tập test (VNĐ)",
      age: 32,
      address: results[0]?.RMSETest_LightGBM,
    },
    {
      key: "5",
      name: "Mean Squared Error trên tập test (VNĐ^2)",
      age: 32,
      address: results[0]?.MSETest_LightGBM,
    },
  ];
  const data5 = [
    {
      key: "1",
      name: results[0]?.GiaThucTeCuoi_LightGBM?.Ngay[0],
      address: results[0]?.GiaThucTeCuoi_LightGBM?.GiaNgayTruoc[0],
      age: results[0]?.GiaThucTeCuoi_LightGBM?.GiaDuDoan[0]
      
    },
    
  ];
  const handleStockChange = (value) => {
    setSelectedStock(value);
  };

  const handleAlgorithmChange = (value) => {
    setSelectedAlgorithm(value);
  };

  const handleParameterChange = (value) => {
    setSelectedParameter(value);
  };

  const handleSearch = async () => {
    setLoading(true);
    console.log(selectedStock, selectedAlgorithm, selectedParameter);
    await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        stock_code: selectedStock,
        algorithm: selectedAlgorithm,
        parameter: selectedParameter,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        // console.log("data", data); // Kiểm tra dữ liệu trả về từ API
        setResults(data);
        // console.log("setResults(data.results);",setResults(data))
        // setChartImage(data.plot);
        // console.log("setChartImage(data.plot)",setChartImage(data.plot))
      })
      .catch((error) => {
        console.error("Error:", error);
      })
      .finally(() => {
        setLoading(false);
      });
  };
  useEffect(() => {
    console.log("results", results);
  }, [results]);

  return (
    <div className="container mx-auto p-8">
      <h1 className="text-5xl font-bold text-center mb-8 text-gray-600 font-serif">
        Stock Price Analysis Dashboard
      </h1>
      <div className="flex justify-center space-x-4">
        <Select
          style={{ width: 200 }}
          placeholder="Chọn thuật toán"
          value={selectedAlgorithm}
          onChange={handleAlgorithmChange}
        >
          <Option value="xgboots">XGBoost</Option>
          <Option value="lightgbm">LightGBM</Option>
          <Option value="lstm">LSTM</Option>
        </Select>

        <Select
          style={{ width: 200 }}
          placeholder="Chọn mã cổ phiếu"
          value={selectedStock}
          onChange={handleStockChange}
        >
          {stockOptions.map((symbol) => (
            <Option key={symbol} value={symbol}>
              {symbol}
            </Option>
          ))}
        </Select>

        <Select
          style={{ width: 200 }}
          placeholder="Chọn tham số"
          value={selectedParameter}
          onChange={handleParameterChange}
        >
          <Option value="default">Tham số mặc định</Option>
          <Option value="custom">Tinh chỉnh tham số</Option>
        </Select>

        <Button type="primary" icon={<SearchOutlined />} onClick={handleSearch}>
          Search
        </Button>
      </div>

      <div className="mt-8">
        {loading ? (
          <Spin size="large" />
        ) : results && results.length > 0 ? (
          <div>
            <div className="flex flex-col justify-around">
              {selectedAlgorithm === "xgboots" && (
                <>
                  <div className="">
                    <h3 className="font-bold text-3xl text-gray-600 font-serif">
                      XGBoost
                    </h3>
                    <Divider>Bảng chỉ số đánh giá các phương pháp trên tập train và tập test cho mã chứng khoáng  {results[0]?.MaChungKhoan}</Divider>
                    <Table bordered  columns={columns} dataSource={data} size="small" />
                    <Table bordered   columns={columns1} dataSource={data1} size="small" />
                    <Divider>Kết quả dự đoán cho ngày tiếp theo cho mã chứng khoáng  {results[0]?.MaChungKhoan}</Divider>
                    <Table bordered pagination={false} columns={columns2} dataSource={data2} size="small" />
                  </div>
                  <div className="flex justify-center mt-1">
                    {results[0].chart && (
                      <img
                        src={`data:image/png;base64,${results[0].chart}`}
                        alt="Stock Price Prediction"
                      />
                    )}
                  </div>
                </>
              )}
            </div>
            <div className="flex flex-col justify-around">
              {selectedAlgorithm === "lightgbm" && (
                <>
                  <div className="">
                    <h3 className="font-bold text-3xl text-gray-600 font-serif">
                      LightGBM
                    </h3>
                    <Divider>Bảng chỉ số đánh giá các phương pháp trên tập train và tập test cho mã chứng khoáng  {results[0]?.MaChungKhoan}</Divider>
                    <Table bordered  columns={columns3} dataSource={data3} size="small" />
                    <Table bordered  columns={columns1} dataSource={data4} size="small" />
                    <Divider>Kết quả dự đoán cho ngày tiếp theo cho mã chứng khoáng  {results[0]?.MaChungKhoan}</Divider>
                    <Table bordered pagination={false} columns={columns2} dataSource={data5} size="small" />
                  </div>
                  <div className="flex justify-center mt-8">
                    {results[0].chart && (
                      <img
                        src={`data:image/png;base64,${results[0].chart}`}
                        alt="Stock Price Prediction"
                      />
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        ) : (
          <p className="text-xl text-gray-500 text-center">
            Không có dữ liệu để hiển thị
          </p>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
