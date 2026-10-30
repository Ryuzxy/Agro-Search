import { useState } from 'react'

const API = '/api'

function App() {
  const [mode, setMode] = useState('search')
  const [query, setQuery] = useState('')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [results, setResults] = useState(null)
  const [classification, setClassification] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    const f = e.target.files[0]
    if (f) {
      setFile(f)
      setPreview(URL.createObjectURL(f))
    }
  }

  const handleSearch = async () => {
    if (!query.trim()) return
    setLoading(true); setError(null); setClassification(null)
    try {
      const res = await fetch(`${API}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_n: 5 }),
      })
      const data = await res.json()
      setResults(data.results)
    } catch (err) {
      setError('Gagal menghubungi server. Pastikan backend berjalan.')
    }
    setLoading(false)
  }

  const handleClassify = async () => {
    if (!file) return
    setLoading(true); setError(null); setResults(null)
    const formData = new FormData()
    formData.append('file', file)
    try {
      const res = await fetch(`${API}/classify`, { method: 'POST', body: formData })
      const data = await res.json()
      if (res.ok) setClassification(data)
      else setError(data.detail || 'Klasifikasi gagal')
    } catch (err) {
      setError('Gagal menghubungi server.')
    }
    setLoading(false)
  }

  const handleHybrid = async () => {
    if (!file) return
    setLoading(true); setError(null)
    const formData = new FormData()
    formData.append('file', file)
    formData.append('top_n', '5')
    try {
      const res = await fetch(`${API}/classify-and-search`, { method: 'POST', body: formData })
      const data = await res.json()
      if (res.ok) {
        setClassification(data.classification)
        setResults(data.results)
      } else setError(data.detail || 'Proses gagal')
    } catch (err) {
      setError('Gagal menghubungi server.')
    }
    setLoading(false)
  }

  const resetState = () => {
    setResults(null); setClassification(null); setError(null)
    setFile(null); setPreview(null); setQuery('')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100">
      <header className="bg-green-800 text-white shadow-lg">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold tracking-tight">AGRO-SEARCH</h1>
          <p className="text-green-200 mt-1">Mesin Pencari Dokumen Solusi Penyakit Tanaman Hortikultura</p>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        <div className="flex gap-2 mb-6">
          {[
            { id: 'search', label: 'Pencarian Teks', desc: 'Cari berdasarkan kata kunci' },
            { id: 'upload', label: 'Klasifikasi Gambar', desc: 'Upload foto daun' },
            { id: 'hybrid', label: 'Hybrid (Foto + Pencarian)', desc: 'Foto > Diagnosis > Dokumen' },
          ].map((tab) => (
            <button key={tab.id} onClick={() => { setMode(tab.id); resetState() }}
              className={`flex-1 p-4 rounded-xl text-left transition-all ${
                mode === tab.id ? 'bg-green-700 text-white shadow-md' : 'bg-white text-gray-700 hover:bg-green-50 border border-gray-200'
              }`}>
              <div className="font-semibold">{tab.label}</div>
              <div className={`text-sm mt-1 ${mode === tab.id ? 'text-green-200' : 'text-gray-500'}`}>{tab.desc}</div>
            </button>
          ))}
        </div>

        <div className="bg-white rounded-xl shadow-md p-6 mb-6">
          {mode === 'search' && (
            <div className="flex gap-3">
              <input type="text" value={query} onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Contoh: layu fusarium cabai, busuk daun kentang..."
                className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent outline-none" />
              <button onClick={handleSearch} disabled={loading}
                className="px-6 py-3 bg-green-700 text-white rounded-lg hover:bg-green-800 disabled:opacity-50 font-medium">
                {loading ? 'Mencari...' : 'Cari'}
              </button>
            </div>
          )}
          {(mode === 'upload' || mode === 'hybrid') && (
            <div>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-green-500 transition-colors">
                <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" id="file-upload" />
                <label htmlFor="file-upload" className="cursor-pointer">
                  {preview ? (
                    <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-lg" />
                  ) : (
                    <div>
                      <p className="text-4xl mb-2">&#128247;</p>
                      <p className="text-gray-600">Klik untuk upload foto daun tanaman</p>
                      <p className="text-gray-400 text-sm mt-1">JPG, PNG (maks 10MB)</p>
                    </div>
                  )}
                </label>
              </div>
              {file && (
                <button onClick={mode === 'hybrid' ? handleHybrid : handleClassify} disabled={loading}
                  className="mt-4 w-full px-6 py-3 bg-green-700 text-white rounded-lg hover:bg-green-800 disabled:opacity-50 font-medium">
                  {loading ? 'Memproses...' : mode === 'hybrid' ? 'Klasifikasi dan Cari Dokumen' : 'Klasifikasi Penyakit'}
                </button>
              )}
            </div>
          )}
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-6 text-red-700">{error}</div>
        )}

        {classification && (
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-6 mb-6">
            <h3 className="font-bold text-blue-800 text-lg mb-2">Hasil Klasifikasi CNN</h3>
            <div className="grid grid-cols-2 gap-4">
              <div><span className="text-blue-600 text-sm">Kelas</span><p className="font-semibold text-gray-800">{classification.class_name}</p></div>
              <div><span className="text-blue-600 text-sm">Confidence</span><p className="font-semibold text-gray-800">{(classification.confidence * 100).toFixed(1)}%</p></div>
              <div><span className="text-blue-600 text-sm">Query Otomatis</span><p className="font-semibold text-gray-800">{classification.query}</p></div>
              <div><span className="text-blue-600 text-sm">Status</span><p className={`font-semibold ${classification.is_healthy ? 'text-green-600' : 'text-red-600'}`}>{classification.is_healthy ? 'Sehat' : 'Terinfeksi'}</p></div>
            </div>
          </div>
        )}

        {results && results.length > 0 && (
          <div>
            <h3 className="font-bold text-gray-800 text-lg mb-4">Dokumen Relevan ({results.length} hasil)</h3>
            <div className="space-y-4">
              {results.map((doc, i) => (
                <div key={doc.id_dokumen} className="bg-white rounded-xl shadow-md p-6 border-l-4 border-green-600">
                  <div className="flex justify-between items-start mb-3">
                    <h4 className="font-bold text-green-800 text-lg">#{i + 1} {doc.judul_penyakit}</h4>
                    {doc.skor_relevansi != null && (
                      <span className="bg-green-100 text-green-800 text-sm px-3 py-1 rounded-full font-medium">Skor: {doc.skor_relevansi}</span>
                    )}
                  </div>
                  <div className="space-y-3 text-sm">
                    <div><span className="font-semibold text-gray-600">Patogen: </span><span className="text-gray-800">{doc.penyebab_patogen}</span></div>
                    <div><span className="font-semibold text-gray-600">Gejala Visual: </span><span className="text-gray-800">{doc.gejala_visual}</span></div>
                    <div><span className="font-semibold text-gray-600">Pestisida: </span><span className="text-gray-800">{doc.rekomendasi_pestisida}</span></div>
                    <div><span className="font-semibold text-gray-600">Langkah Penanganan:</span><p className="text-gray-800 mt-1 whitespace-pre-line">{doc.langkah_penanganan}</p></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {results && results.length === 0 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6 text-center text-yellow-700">
            Tidak ditemukan dokumen yang relevan. Coba kata kunci lain.
          </div>
        )}
      </main>

      <footer className="bg-green-900 text-green-300 py-6 mt-12">
        <div className="max-w-6xl mx-auto px-4 text-center text-sm">
          <p>AGRO-SEARCH 2026 - Program Studi Informatika, Fakultas Ilmu Komputer, Universitas Jember</p>
          <p className="mt-1 text-green-400">Yuri Wicaksono - Aditya Satya Lencana - Michael Yudananta - Dian Isnariadi Widyaswara</p>
        </div>
      </footer>
    </div>
  )
}

export default App
