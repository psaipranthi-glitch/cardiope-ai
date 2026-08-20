import { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";

const API = "https://cardiope-ai.onrender.com";

function App() {
  /* =========================================================
     AUTH
  ========================================================= */
  const [token, setToken] = useState(
    localStorage.getItem("token")
  );

  const [email, setEmail] = useState(
    localStorage.getItem("email") ||
    "doctor@test.com"
  );

  const [password, setPassword] = useState(
    "DoctorTest123"
  );

  const [loginLoading, setLoginLoading] =
    useState(false);

  const [loginMessage, setLoginMessage] =
    useState("");

  /* =========================================================
     SYSTEM
  ========================================================= */
  const [backendOnline, setBackendOnline] =
    useState(false);

  const [predictionLoading, setPredictionLoading] =
    useState(false);

  const [predictionError, setPredictionError] =
    useState("");

  /* =========================================================
     PATIENTS
  ========================================================= */
  const [patients, setPatients] =
    useState([]);

  const [selectedPatient, setSelectedPatient] =
    useState(null);

  const [patientLoading, setPatientLoading] =
    useState(false);

  const [showPatientForm, setShowPatientForm] =
    useState(false);

  const [newPatient, setNewPatient] = useState({
    name: "",
    age: "",
    gender: "",
    phone: "",
    blood_group: "",
  });

  /* =========================================================
     CLINICAL
  ========================================================= */
  const [clinicalText, setClinicalText] =
    useState("");

  /* =========================================================
     X-RAY
  ========================================================= */
  const [xrayFile, setXrayFile] =
    useState(null);

  const [xrayPreview, setXrayPreview] =
    useState("");

  /* =========================================================
     ECG
  ========================================================= */
  const [ecgFile, setEcgFile] =
    useState(null);

  const [ecgPreview, setEcgPreview] =
    useState("");

  /* =========================================================
     RESULT
  ========================================================= */
  const [result, setResult] =
    useState(null);

  /* =========================================================
     HISTORY
  ========================================================= */
  const [predictionHistory, setPredictionHistory] =
    useState([]);

  const [historyLoading, setHistoryLoading] =
    useState(false);

  /* =========================================================
     BACKEND HEALTH
  ========================================================= */
  useEffect(() => {
    checkBackend();

    const interval = setInterval(() => {
      checkBackend();
    }, 10000);

    return () => {
      clearInterval(interval);
    };
  }, []);

  /* =========================================================
     CHECK BACKEND
  ========================================================= */
  const checkBackend = async () => {
    try {
      await axios.get(
        `${API}/health`,
        {
          timeout: 5000,
        }
      );

      setBackendOnline(true);
    } catch {
      setBackendOnline(false);
    }
  };

  /* =========================================================
     AUTH HEADERS
  ========================================================= */
  const getAuthHeaders = () => {
    const currentToken =
      localStorage.getItem("token");

    return {
      Authorization:
        `Bearer ${currentToken}`,
    };
  };

  /* =========================================================
     CLEAR AUTH
  ========================================================= */
  const clearAuthentication = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user_id");
    localStorage.removeItem("role");
    localStorage.removeItem("email");

    setToken(null);
    setPredictionLoading(false);
    setPredictionError(
      "Your session has expired. Please login again."
    );
    setResult(null);
  };

  /* =========================================================
     LOGIN
  ========================================================= */
  const login = async (e) => {
    e.preventDefault();
    setLoginLoading(true);
    setLoginMessage("");

    try {
      const response = await axios.post(
        `${API}/auth/login`,
        {
          email,
          password,
        },
        {
          timeout: 10000,
        }
      );

      const data = response.data;

      const receivedToken =
        data.token ||
        data.access_token;

      if (!receivedToken) {
        throw new Error(
          "Login succeeded but backend did not return an authentication token."
        );
      }

      localStorage.setItem(
        "token",
        receivedToken
      );

      if (data.user_id) {
        localStorage.setItem(
          "user_id",
          data.user_id
        );
      }

      if (data.role) {
        localStorage.setItem(
          "role",
          data.role
        );
      }

      localStorage.setItem(
        "email",
        email
      );

      setToken(receivedToken);

      setLoginMessage(
        "Login successful"
      );

      setPredictionError("");

      checkBackend();
    } catch (error) {
      console.error(
        "Login error:",
        error
      );

      setLoginMessage(
        error.response?.data?.detail ||
        error.response?.data?.message ||
        error.message ||
        "Login failed"
      );
    } finally {
      setLoginLoading(false);
    }
  };

  /* =========================================================
     LOGOUT
  ========================================================= */
  const logout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user_id");
    localStorage.removeItem("role");
    localStorage.removeItem("email");

    setToken(null);
    setPatients([]);
    setSelectedPatient(null);
    setClinicalText("");
    setXrayFile(null);
    setXrayPreview("");
    setEcgFile(null);
    setEcgPreview("");
    setPredictionError("");
    setResult(null);
    setPredictionHistory([]);
  };

  /* =========================================================
     LOAD PATIENTS
  ========================================================= */
  const loadPatients = async () => {
    const currentToken =
      localStorage.getItem("token");

    if (!currentToken) {
      return;
    }

    try {
      setPatientLoading(true);

      const response = await axios.get(
        `${API}/patients/`,
        {
          headers: getAuthHeaders(),
        }
      );

      const data = response.data;
      let patientList = [];

      if (Array.isArray(data)) {
        patientList = data;
      } else if (Array.isArray(data?.patients)) {
        patientList = data.patients;
      } else if (Array.isArray(data?.data)) {
        patientList = data.data;
      }

      setPatients(patientList);
    } catch (error) {
      console.error(
        "Patient loading error:",
        error
      );

      setPatients([]);

      if (
        error.response?.status === 401
      ) {
        clearAuthentication();
      }
    } finally {
      setPatientLoading(false);
    }
  };

  useEffect(() => {
    if (token) {
      loadPatients();
    }
  }, [token]);

  /* =========================================================
     ADD PATIENT
  ========================================================= */
  const addPatient = async (e) => {
    e.preventDefault();

    setPatientLoading(true);
    setPredictionError("");

    try {
      const payload = {
        name:
          newPatient.name.trim(),
        age:
          Number(newPatient.age),
        gender:
          newPatient.gender,
        phone:
          newPatient.phone.trim(),
        blood_group:
          newPatient.blood_group,
      };

      const response = await axios.post(
        `${API}/patients/`,
        payload,
        {
          headers: {
            ...getAuthHeaders(),
            "Content-Type":
              "application/json",
          },
        }
      );

      await loadPatients();

      const createdPatient = {
        ...payload,
        patient_id:
          response.data.patient_id,
        doctor_id:
          response.data.doctor_id,
      };

      setSelectedPatient(
        createdPatient
      );

      setNewPatient({
        name: "",
        age: "",
        gender: "",
        phone: "",
        blood_group: "",
      });

      setShowPatientForm(false);
      setPredictionError("");
    } catch (error) {
      console.error(
        "Add patient error:",
        error
      );

      setPredictionError(
        error.response?.data?.detail ||
        "Failed to create patient."
      );
    } finally {
      setPatientLoading(false);
    }
  };

  /* =========================================================
     SELECT PATIENT
  ========================================================= */
  const selectPatient = async (patientId) => {
    if (!patientId) {
      setSelectedPatient(null);
      setPredictionHistory([]);
      return;
    }

    const safePatients = Array.isArray(patients) ? patients : [];
    const patient = safePatients.find(
      (item) =>
        item.patient_id === patientId
    );

    if (!patient) {
      return;
    }

    setSelectedPatient(patient);
    setClinicalText("");
    setXrayFile(null);
    setXrayPreview("");
    setEcgFile(null);
    setEcgPreview("");
    setResult(null);
    setPredictionError("");

    await loadPredictionHistory(
      patient.patient_id
    );
  };

  /* =========================================================
     LOAD HISTORY
  ========================================================= */
  const loadPredictionHistory = async (patientId) => {
    if (!patientId) {
      return;
    }

    try {
      setHistoryLoading(true);

      const response = await axios.get(
        `${API}/patients/${patientId}/predictions`,
        {
          headers: getAuthHeaders(),
          // Custom status validator to keep 404 from throwing an Unhandled Exception
          validateStatus: (status) =>
            (status >= 200 && status < 300) || status === 404,
        }
      );

      // Gracefully handle 404 when patient history is empty
      if (response.status === 404) {
        setPredictionHistory([]);
        return;
      }

      const data = response.data;
      if (Array.isArray(data)) {
        setPredictionHistory(data);
      } else if (Array.isArray(data?.predictions)) {
        setPredictionHistory(data.predictions);
      } else {
        setPredictionHistory([]);
      }
    } catch (error) {
      if (error.response?.status === 401) {
        clearAuthentication();
      } else {
        setPredictionHistory([]);
      }
    } finally {
      setHistoryLoading(false);
    }
  };

  /* =========================================================
     X-RAY
  ========================================================= */
  const handleXrayChange = (e) => {
    const file =
      e.target.files?.[0];

    if (!file) {
      return;
    }

    const allowedTypes = [
      "image/png",
      "image/jpeg",
      "image/jpg",
    ];

    if (
      !allowedTypes.includes(
        file.type
      )
    ) {
      setPredictionError(
        "Only PNG, JPG or JPEG X-ray images are supported."
      );
      return;
    }

    setXrayFile(file);
    setPredictionError("");
    setResult(null);

    const previewURL =
      URL.createObjectURL(file);

    setXrayPreview(
      previewURL
    );
  };

  /* =========================================================
     ECG
  ========================================================= */
  const handleEcgChange = (e) => {
    const file =
      e.target.files?.[0];

    if (!file) {
      return;
    }

    const allowedTypes = [
      "image/png",
      "image/jpeg",
      "image/jpg",
      "application/pdf",
    ];

    if (
      !allowedTypes.includes(
        file.type
      )
    ) {
      setPredictionError(
        "ECG must be PNG, JPG, JPEG or PDF."
      );
      return;
    }

    setEcgFile(file);
    setPredictionError("");
    setResult(null);

    if (
      file.type.startsWith(
        "image/"
      )
    ) {
      const previewURL =
        URL.createObjectURL(file);

      setEcgPreview(
        previewURL
      );
    } else {
      setEcgPreview("");
    }
  };

  /* =========================================================
     SAFE PROBABILITY
  ========================================================= */
  const getSafeProbability = (responseData) => {
    if (!responseData) {
      return null;
    }

    let data =
      responseData.result ||
      responseData;

    let probability = null;

    if (
      data &&
      data.abnormal_probability !==
        undefined &&
      data.abnormal_probability !==
        null
    ) {
      probability =
        Number(
          data.abnormal_probability
        );
    }

    if (
      !Number.isFinite(
        probability
      ) &&
      data &&
      data.risk_percentage !==
        undefined &&
      data.risk_percentage !==
        null
    ) {
      const percentage =
        Number(
          data.risk_percentage
        );

      if (
        Number.isFinite(
          percentage
        )
      ) {
        probability =
          percentage / 100;
      }
    }

    if (
      !Number.isFinite(
        probability
      ) &&
      data?.probability !==
        undefined
    ) {
      probability =
        Number(
          data.probability
        );
    }

    if (
      !Number.isFinite(
        probability
      ) &&
      data?.risk_probability !==
        undefined
    ) {
      probability =
        Number(
          data.risk_probability
        );
    }

    if (
      Number.isFinite(
        probability
      ) &&
      probability > 1 &&
      probability <= 100
    ) {
      probability =
        probability / 100;
    }

    if (
      !Number.isFinite(
        probability
      ) ||
      probability < 0 ||
      probability > 1
    ) {
      return null;
    }

    return probability;
  };

  /* =========================================================
     ASSESSMENT
  ========================================================= */
  const getAssessment = (
    responseData,
    probability
  ) => {
    const data =
      responseData?.result ||
      responseData ||
      {};

    let assessment =
      data.assessment;

    if (
      typeof assessment !==
        "string" ||
      !assessment.trim()
    ) {
      assessment =
        probability >= 0.5
          ? "HIGH RISK"
          : "LOW RISK";
    }

    return assessment.trim();
  };

  /* =========================================================
   RUN MULTIMODAL PREDICTION
========================================================= */
const runPrediction = async () => {
  setPredictionError("");
  setResult(null);
  setPredictionLoading(true);

  try {
    if (!selectedPatient) {
      throw new Error("Please select a patient first.");
    }

    if (!clinicalText.trim()) {
      throw new Error("Please enter clinical information.");
    }

    if (!xrayFile) {
      throw new Error("Please upload a chest X-ray.");
    }

    if (!ecgFile) {
      throw new Error("Please upload an ECG.");
    }

    const currentToken = localStorage.getItem("token");

    if (!currentToken) {
      clearAuthentication();
      return;
    }

    if (!backendOnline) {
      throw new Error(
        "Backend is offline. Please start FastAPI."
      );
    }

    const formData = new FormData();

    /* PATIENT */
    formData.append(
      "patient_id",
      selectedPatient.patient_id
    );

    /* CLINICAL */
    formData.append(
      "clinical_text",
      clinicalText.trim()
    );

    /* CHEST X-RAY */
    formData.append(
      "xray",
      xrayFile
    );

    /* ECG */
    formData.append(
      "ecg",
      ecgFile
    );

    const response = await axios.post(
      `${API}/predict/`,
      formData,
      {
        headers: {
          Authorization: `Bearer ${currentToken}`,
        },
        timeout: 120000,
      }
    );

    console.log(
      "CARDIOPE-AI RESPONSE:",
      response.data
    );

    const data =
      response.data?.result ||
      response.data ||
      {};

    /* =====================================================
       PROBABILITY
    ===================================================== */

    let probability = null;

    if (
      data.abnormal_probability !== undefined &&
      data.abnormal_probability !== null
    ) {
      probability = Number(
        data.abnormal_probability
      );
    }

    if (
      !Number.isFinite(probability) &&
      data.risk_probability !== undefined &&
      data.risk_probability !== null
    ) {
      probability = Number(
        data.risk_probability
      );
    }

    if (
      !Number.isFinite(probability) &&
      data.probability !== undefined &&
      data.probability !== null
    ) {
      probability = Number(
        data.probability
      );
    }

    if (
      !Number.isFinite(probability) &&
      data.risk_percentage !== undefined &&
      data.risk_percentage !== null
    ) {
      probability =
        Number(data.risk_percentage) / 100;
    }

    /* Convert percentage → probability */
    if (
      Number.isFinite(probability) &&
      probability > 1 &&
      probability <= 100
    ) {
      probability =
        probability / 100;
    }

    if (
      !Number.isFinite(probability) ||
      probability < 0 ||
      probability > 1
    ) {
      throw new Error(
        "Backend returned an invalid risk probability."
      );
    }

    const percentage =
      Math.max(
        0,
        Math.min(
          100,
          probability * 100
        )
      );

    /* =====================================================
       ASSESSMENT
    ===================================================== */

    const assessment =
      typeof data.assessment === "string" &&
      data.assessment.trim()
        ? data.assessment.trim()
        : probability >= 0.5
        ? "HIGH RISK"
        : "LOW RISK";

    /* =====================================================
       MODALITY RESULTS
    ===================================================== */

    const finalResult = {
      probability,
      percentage,
      assessment,

      prediction_id:
        response.data?.prediction_id ||
        data.prediction_id,

      ecg_result:
        data.ecg_result ||
        data.ecg_prediction ||
        data.ecg_assessment ||
        null,

      xray_result:
        data.xray_result ||
        data.xray_prediction ||
        data.xray_assessment ||
        null,

      clinical_result:
        data.clinical_result ||
        data.clinical_prediction ||
        null,

      fusion_result:
        data.fusion_result ||
        data.fused_prediction ||
        data.final_prediction ||
        assessment,
    };

    console.log(
      "FINAL CARDIOPE-AI RESULT:",
      finalResult
    );

    setResult(finalResult);

    /* Refresh patient prediction history */
    await loadPredictionHistory(
      selectedPatient.patient_id
    );

  } catch (error) {
    console.error(
      "Prediction error:",
      error
    );

    if (
      error.response?.status === 401
    ) {
      clearAuthentication();
      return;
    }

    let message =
      "Prediction failed.";

    if (
      error.response?.status === 422
    ) {
      const detail =
        error.response?.data?.detail;

      if (Array.isArray(detail)) {
        message = detail
          .map((item) => {
            const field =
              Array.isArray(item.loc)
                ? item.loc[
                    item.loc.length - 1
                  ]
                : "Field";

            return `${field}: ${
              item.msg ||
              "Invalid request"
            }`;
          })
          .join(", ");
      } else if (
        typeof detail === "string"
      ) {
        message = detail;
      }
    } else if (
      error.response?.data?.detail
    ) {
      message =
        typeof error.response.data.detail ===
        "string"
          ? error.response.data.detail
          : JSON.stringify(
              error.response.data.detail
            );
    } else if (error.message) {
      message = error.message;
    }

    setPredictionError(message);

  } finally {
    setPredictionLoading(false);
  }
};

  /* =========================================================
     LOGIN PAGE
  ========================================================= */
  if (!token) {
    return (
      <div className="app">
        <div className="login-page">
          <div className="ecg-background">
            <svg
              viewBox="0 0 1200 220"
              preserveAspectRatio="none"
            >
              <path
                className="ecg-line"
                d="
                  M0 110
                  L180 110
                  L210 110
                  L225 85
                  L240 135
                  L255 110
                  L390 110
                  L420 110
                  L435 65
                  L450 155
                  L465 110
                  L610 110
                  L640 110
                  L655 90
                  L670 130
                  L685 110
                  L820 110
                  L850 110
                  L865 55
                  L880 165
                  L895 110
                  L1040 110
                  L1070 110
                  L1085 88
                  L1100 132
                  L1115 110
                  L1200 110
                "
              />
            </svg>
          </div>

          <div className="login-card">
            <div className="brand-mark">
              <div className="heart-symbol">
                ♥
              </div>
            </div>

            <h1>
              Cardio<span>PE-AI</span>
            </h1>

            <p className="subtitle">
              Multimodal Cardiac Intelligence
            </p>

            <div className="system-status">
              <span
                className={`status-dot ${
                  backendOnline
                    ? "online"
                    : "offline"
                }`}
              />
              {backendOnline
                ? "SYSTEM ONLINE"
                : "CONNECTING TO SYSTEM"}
            </div>

            <form onSubmit={login}>
              <div className="input-group">
                <label>
                  EMAIL
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) =>
                    setEmail(
                      e.target.value
                    )
                  }
                  placeholder="Doctor email"
                  required
                />
              </div>

              <div className="input-group">
                <label>
                  PASSWORD
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) =>
                    setPassword(
                      e.target.value
                    )
                  }
                  placeholder="Password"
                  required
                />
              </div>

              <button
                className="login-button"
                type="submit"
                disabled={loginLoading}
              >
                {loginLoading
                  ? "AUTHENTICATING..."
                  : "LOGIN"}
              </button>
            </form>

            {loginMessage && (
              <div className="login-message">
                {loginMessage}
              </div>
            )}

            <div className="security-text">
              Secure Doctor Authentication
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* =========================================================
     DASHBOARD
  ========================================================= */
  return (
    <div className="dashboard">
      <div className="dashboard-ecg">
        <svg
          viewBox="0 0 1200 180"
          preserveAspectRatio="none"
        >
          <path
            className="ecg-line"
            d="
              M0 90
              L180 90
              L210 90
              L225 65
              L240 115
              L255 90
              L390 90
              L420 90
              L435 45
              L450 135
              L465 90
              L610 90
              L640 90
              L655 70
              L670 110
              L685 90
              L820 90
              L850 90
              L865 35
              L880 145
              L895 90
              L1040 90
              L1070 90
              L1085 68
              L1100 112
              L1115 90
              L1200 90
            "
          />
        </svg>
      </div>

      {/* =====================================================
         TOPBAR
      ===================================================== */}
      <header className="topbar">
        <div className="brand">
          <div className="mini-heart">
            ♥
          </div>
          <div>
            <div className="brand-title">
              Cardio<span>PE-AI</span>
            </div>
            <div className="brand-subtitle">
              MULTIMODAL CARDIAC INTELLIGENCE
            </div>
          </div>
        </div>

        <div className="topbar-actions">
          {selectedPatient && (
            <div className="current-patient-badge">
              <span>
                PATIENT
              </span>
              <strong>
                {selectedPatient.name}
              </strong>
            </div>
          )}

          <button
            className="logout-button"
            onClick={logout}
          >
            Logout
          </button>
        </div>
      </header>

      <main className="dashboard-content">
        {/* ===================================================
           HEADING
        =================================================== */}
        <div className="dashboard-heading">
          <div>
            <p className="eyebrow">
              CARDIOPE-AI / CLINICAL ENGINE
            </p>
            <h2>
              Cardiac Intelligence Dashboard
            </h2>
            <p>
              Multimodal cardiac risk assessment
              using ECG, clinical NLP and computer
              vision.
            </p>
          </div>

          <div className="api-badge">
            <span
              className={`status-dot ${
                backendOnline
                  ? "online"
                  : "offline"
              }`}
            />
            {backendOnline
              ? "SYSTEM ONLINE"
              : "BACKEND OFFLINE"}
          </div>
        </div>

        {/* ===================================================
           PATIENT MANAGEMENT
        =================================================== */}
        <section className="patient-management panel">
          <div className="panel-number">
            00
          </div>

          <div className="panel-header">
            <div>
              <p className="panel-label">
                PATIENT MANAGEMENT
              </p>
              <h3>
                Patient Records
              </h3>
            </div>

            <span className="online-badge">
              DATABASE
            </span>
          </div>

          <div className="patient-selector">
            <select
              value={
                selectedPatient?.patient_id ||
                ""
              }
              onChange={(e) =>
                selectPatient(
                  e.target.value
                )
              }
            >
              <option value="">
                Select a patient
              </option>

              {(Array.isArray(patients) ? patients : []).map(
                (patient) => (
                  <option
                    key={
                      patient.patient_id
                    }
                    value={
                      patient.patient_id
                    }
                  >
                    {patient.name}
                    {" — "}
                    {patient.age} yrs
                  </option>
                )
              )}
            </select>

            <button
              className="secondary-button"
              onClick={() =>
                setShowPatientForm(
                  !showPatientForm
                )
              }
            >
              {showPatientForm
                ? "Cancel"
                : "+ Add Patient"}
            </button>
          </div>

          {/* =================================================
             ADD PATIENT
          ================================================= */}
          {showPatientForm && (
            <form
              className="patient-form"
              onSubmit={addPatient}
            >
              <div className="form-field">
                <label>
                  FULL NAME
                </label>
                <input
                  type="text"
                  value={
                    newPatient.name
                  }
                  onChange={(e) =>
                    setNewPatient({
                      ...newPatient,
                      name:
                        e.target.value,
                    })
                  }
                  placeholder="Patient name"
                  required
                />
              </div>

              <div className="form-field">
                <label>
                  AGE
                </label>
                <input
                  type="number"
                  min="1"
                  max="120"
                  value={
                    newPatient.age
                  }
                  onChange={(e) =>
                    setNewPatient({
                      ...newPatient,
                      age:
                        e.target.value,
                    })
                  }
                  placeholder="Age"
                  required
                />
              </div>

              <div className="form-field">
                <label>
                  GENDER
                </label>
                <select
                  value={
                    newPatient.gender
                  }
                  onChange={(e) =>
                    setNewPatient({
                      ...newPatient,
                      gender:
                        e.target.value,
                    })
                  }
                  required
                >
                  <option value="">
                    Select gender
                  </option>
                  <option value="Male">
                    Male
                  </option>
                  <option value="Female">
                    Female
                  </option>
                  <option value="Other">
                    Other
                  </option>
                </select>
              </div>

              <div className="form-field">
                <label>
                  BLOOD GROUP
                </label>
                <select
                  value={
                    newPatient.blood_group
                  }
                  onChange={(e) =>
                    setNewPatient({
                      ...newPatient,
                      blood_group:
                        e.target.value,
                    })
                  }
                  required
                >
                  <option value="">
                    Blood group
                  </option>
                  <option value="A+">
                    A+
                  </option>
                  <option value="A-">
                    A-
                  </option>
                  <option value="B+">
                    B+
                  </option>
                  <option value="B-">
                    B-
                  </option>
                  <option value="AB+">
                    AB+
                  </option>
                  <option value="AB-">
                    AB-
                  </option>
                  <option value="O+">
                    O+
                  </option>
                  <option value="O-">
                    O-
                  </option>
                </select>
              </div>

              <div className="form-field">
                <label>
                  PHONE
                </label>
                <input
                  type="tel"
                  value={
                    newPatient.phone
                  }
                  onChange={(e) =>
                    setNewPatient({
                      ...newPatient,
                      phone:
                        e.target.value,
                    })
                  }
                  placeholder="Phone number"
                  required
                />
              </div>

              <button
                className="run-button"
                type="submit"
                disabled={
                  patientLoading
                }
              >
                {patientLoading
                  ? "CREATING PATIENT..."
                  : "CREATE PATIENT"}
              </button>
            </form>
          )}

          {/* =================================================
             SELECTED PATIENT
          ================================================= */}
          {selectedPatient && (
            <div className="selected-patient">
              <div>
                <span>
                  NAME
                </span>
                <strong>
                  {selectedPatient.name}
                </strong>
              </div>

              <div>
                <span>
                  AGE
                </span>
                <strong>
                  {selectedPatient.age}
                </strong>
              </div>

              <div>
                <span>
                  GENDER
                </span>
                <strong>
                  {selectedPatient.gender}
                </strong>
              </div>

              <div>
                <span>
                  BLOOD GROUP
                </span>
                <strong>
                  {selectedPatient.blood_group}
                </strong>
              </div>

              <div>
                <span>
                  PHONE
                </span>
                <strong>
                  {selectedPatient.phone}
                </strong>
              </div>
            </div>
          )}
        </section>

        {/* ===================================================
           INPUT GRID
        =================================================== */}
        <div className="input-grid">
          {/* =================================================
             CLINICAL
          ================================================= */}
          <section className="panel">
            <div className="panel-number">
              01
            </div>

            <div className="panel-header">
              <div>
                <p className="panel-label">
                  PATIENT INFORMATION
                </p>
                <h3>
                  Clinical Information
                </h3>
              </div>

              <span className="ready-badge">
                READY
              </span>
            </div>

            <div className="clinical-section">
              <textarea
                value={clinicalText}
                onChange={(e) =>
                  setClinicalText(
                    e.target.value
                  )
                }
                disabled={
                  !selectedPatient
                }
                placeholder={
                  selectedPatient
                    ? "Enter patient symptoms, medical history, observations..."
                    : "Select a patient first..."
                }
              />

              <div className="character-count">
                {clinicalText.length}
                {" "}
                characters
              </div>
            </div>
          </section>

          {/* =================================================
             X-RAY
          ================================================= */}
          <section className="panel">
            <div className="panel-number">
              02
            </div>

            <div className="panel-header">
              <div>
                <p className="panel-label">
                  COMPUTER VISION INPUT
                </p>
                <h3>
                  Chest X-Ray
                </h3>
              </div>

              <span className="image-badge">
                IMAGE
              </span>
            </div>

            <label
              className={`upload-area ${
                !selectedPatient
                  ? "upload-disabled"
                  : ""
              }`}
            >
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                onChange={
                  handleXrayChange
                }
                disabled={
                  !selectedPatient
                }
              />

              {xrayPreview ? (
                <img
                  src={
                    xrayPreview
                  }
                  className="xray-preview"
                  alt="Chest X-Ray preview"
                />
              ) : (
                <div className="upload-content">
                  <div className="upload-icon">
                    +
                  </div>
                  <strong>
                    Load Chest X-Ray
                  </strong>
                  <span>
                    PNG, JPG or JPEG
                  </span>
                </div>
              )}
            </label>

            {xrayFile && (
              <div className="file-name">
                {xrayFile.name}
              </div>
            )}
          </section>
        </div>

        {/* ===================================================
           ECG + AI
        =================================================== */}
        <div className="analysis-grid">
          {/* =================================================
             ECG
          ================================================= */}
          <section className="panel">
            <div className="panel-number">
              03
            </div>

            <div className="panel-header">
              <div>
                <p className="panel-label">
                  ELECTROCARDIOGRAM SIGNAL
                </p>
                <h3>
                  ECG Analysis
                </h3>
              </div>

              <span className="ready-badge">
                INPUT
              </span>
            </div>

            <label
              className={`ecg-upload ${
                !selectedPatient
                  ? "upload-disabled"
                  : ""
              }`}
            >
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg,application/pdf"
                onChange={
                  handleEcgChange
                }
                disabled={
                  !selectedPatient
                }
              />

              {ecgPreview ? (
                <img
                  src={
                    ecgPreview
                  }
                  className="ecg-preview-image"
                  alt="ECG preview"
                />
              ) : (
                <div className="ecg-display">
                  <svg
                    viewBox="0 0 900 120"
                    preserveAspectRatio="none"
                  >
                    <path
                      className="ecg-display-line"
                      d="
                        M0 60
                        L100 60
                        L130 60
                        L145 42
                        L160 78
                        L175 60
                        L280 60
                        L310 60
                        L325 30
                        L340 90
                        L355 60
                        L470 60
                        L500 60
                        L515 45
                        L530 75
                        L545 60
                        L660 60
                        L690 60
                        L705 25
                        L720 95
                        L735 60
                        L840 60
                        L870 60
                        L885 45
                        L900 60
                      "
                    />
                  </svg>

                  <div className="ecg-upload-overlay">
                    <strong>
                      {selectedPatient
                        ? "Upload ECG"
                        : "Select patient first"}
                    </strong>
                    <span>
                      PNG, JPG, JPEG or PDF
                    </span>
                  </div>
                </div>
              )}
            </label>

            {ecgFile && (
              <div className="file-name">
                {ecgFile.name}
              </div>
            )}

            <p className="demo-note">
              ECG data is stored with the
              patient prediction record.
            </p>
          </section>

          {/* =================================================
             AI
          ================================================= */}
          <section className="panel">
            <div className="panel-number">
              04
            </div>

            <div className="panel-header">
              <div>
                <p className="panel-label">
                  MULTIMODAL INTELLIGENCE ENGINE
                </p>
                <h3>
                  AI Analysis
                </h3>
              </div>

              <span className="online-badge">
                ONLINE
              </span>
            </div>

            <div className="model-list">
              <div className="model-row">
                <span>
                  ECG CNN
                </span>
                <b>
                  READY
                </b>
              </div>

              <div className="model-row">
                <span>
                  Clinical NLP
                </span>
                <b>
                  READY
                </b>
              </div>

              <div className="model-row">
                <span>
                  Computer Vision
                </span>
                <b>
                  READY
                </b>
              </div>

              <div className="model-row fusion-row">
                <span>
                  CardioFusion
                </span>
                <b>
                  READY
                </b>
              </div>
            </div>

            <button
              className="run-button"
              onClick={
                runPrediction
              }
              disabled={
                predictionLoading ||
                !selectedPatient
              }
            >
              <span>
                {predictionLoading
                  ? "ANALYZING..."
                  : "RUN CARDIOPE-AI"}
              </span>

              {!predictionLoading && (
                <span>
                  →
                </span>
              )}
            </button>
          </section>
        </div>

        {/* ===================================================
           ERROR
        =================================================== */}
        {predictionError && (
          <div className="error-panel">
            {predictionError}
          </div>
        )}

        {/* ===================================================
           RESULT
        =================================================== */}
        {result && (
          <section className="result-panel">
            <div className="result-header">
              <div>
                <p className="eyebrow">
                  CARDIOPE-AI / ANALYSIS COMPLETE
                </p>
                <h2>
                  Cardiac Risk Assessment
                </h2>
              </div>

              <div className="result-status">
                COMPLETE
              </div>
            </div>

            <div className="result-patient">
              <span>
                PATIENT
              </span>
              <strong>
                {selectedPatient.name}
              </strong>
              <span>
                ID
              </span>
              <strong>
                {selectedPatient.patient_id}
              </strong>
            </div>

            <div className="result-content">
              <div className="probability-box">
                <span>
                  ABNORMAL PROBABILITY
                </span>

                <strong>
                  {Number.isFinite(
                    result.percentage
                  )
                    ? `${result.percentage.toFixed(2)}%`
                    : "0.00%"}
                </strong>

                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${
                        Number.isFinite(
                          result.percentage
                        )
                          ? result.percentage
                          : 0
                      }%`,
                    }}
                  />
                </div>
              </div>

              <div
                className={`risk-box ${
                  String(
                    result.assessment
                  )
                    .toUpperCase()
                    .includes("HIGH")
                    ? "high-risk"
                    : "low-risk"
                }`}
              >
                <span>
                  FINAL ASSESSMENT
                </span>

                <strong>
                  {result.assessment}
                </strong>
              </div>
            </div>

            <p className="research-warning">
              Prediction ID:
              {" "}
              {result.prediction_id || "Stored"}
              <br />
              Research prototype only.
              This output is not a medical diagnosis.
            </p>
          </section>
        )}

        {/* ===================================================
           HISTORY
        =================================================== */}
        {selectedPatient && (
          <section className="history-panel">
            <div className="history-header">
              <div>
                <p className="eyebrow">
                  PATIENT / HISTORY
                </p>
                <h2>
                  Prediction History
                </h2>
              </div>

              <button
                className="secondary-button"
                onClick={() =>
                  loadPredictionHistory(
                    selectedPatient.patient_id
                  )
                }
              >
                Refresh
              </button>
            </div>

            {historyLoading ? (
              <div className="history-empty">
                Loading prediction history...
              </div>
            ) : (Array.isArray(predictionHistory) ? predictionHistory : []).length === 0 ? (
              <div className="history-empty">
                No predictions recorded
                for this patient yet.
              </div>
            ) : (
              <div className="history-list">
                {(Array.isArray(predictionHistory) ? predictionHistory : []).map(
                  (prediction) => (
                    <div
                      className="history-row"
                      key={
                        prediction.prediction_id
                      }
                    >
                      <div>
                        <span className="history-label">
                          DATE
                        </span>
                        <strong>
                          {prediction.created_at
                            ? new Date(
                                prediction.created_at
                              ).toLocaleString()
                            : "Unknown"}
                        </strong>
                      </div>

                      <div>
                        <span className="history-label">
                          RISK
                        </span>
                        <strong>
                          {prediction.result
                            ?.risk_percentage
                            ?? 0}
                          %
                        </strong>
                      </div>

                      <div>
                        <span className="history-label">
                          ASSESSMENT
                        </span>
                        <strong
                          className={
                            String(
                              prediction.result
                                ?.assessment
                            )
                              .toUpperCase()
                              .includes(
                                "HIGH"
                              )
                              ? "history-high"
                              : "history-low"
                          }
                        >
                          {
                            prediction.result
                              ?.assessment
                          }
                        </strong>
                      </div>

                      <div>
                        <span className="history-label">
                          ECG
                        </span>
                        <strong>
                          {prediction.ecg_filename
                            ? "UPLOADED"
                            : "NOT PROVIDED"}
                        </strong>
                      </div>
                    </div>
                  )
                )}
              </div>
            )}
          </section>
        )}

        <p className="research-warning dashboard-warning">
          CardioPE-AI is a research prototype.
          Predictions should not be used as a substitute
          for professional medical diagnosis.
        </p>
      </main>
    </div>
  );
}

export default App;