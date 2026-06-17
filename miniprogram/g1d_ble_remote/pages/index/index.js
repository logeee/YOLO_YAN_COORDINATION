const SERVICE_UUID = '6f9d0001-7f70-4f8f-9f25-41f0a7a1b001'
const CONTROL_UUID = '6f9d0002-7f70-4f8f-9f25-41f0a7a1b001'
const HEARTBEAT_MS = 250
const REMOTE_NAME_RE = /^G1D-BLE-RCS-\d{4,6}$/

function normalizeUuid(value) {
  return String(value || '').toLowerCase().replace(/-/g, '')
}

function sameUuid(left, right) {
  return normalizeUuid(left) === normalizeUuid(right)
}

function isG1NamedDevice(device) {
  const name = device.name || device.localName || ''
  return REMOTE_NAME_RE.test(name)
}

function textToArrayBuffer(text) {
  const buffer = new ArrayBuffer(text.length)
  const view = new Uint8Array(buffer)
  for (let i = 0; i < text.length; i += 1) {
    view[i] = text.charCodeAt(i)
  }
  return buffer
}

function stamp() {
  const d = new Date()
  const pad = (n, w = 2) => String(n).padStart(w, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`
}

Page({
  data: {
    scanning: false,
    validating: false,
    connected: false,
    deviceId: '',
    deviceName: '',
    serviceId: '',
    controlId: '',
    speedValue: 20,
    speedText: '0.20',
    lastAction: '无',
    activeAction: '',
    candidateDevices: [],
    logs: []
  },

  heartbeatTimer: null,
  holdingAction: '',
  deviceFoundHandlerReady: false,
  connectionHandlerReady: false,
  scanSeen: 0,
  scanFound: 0,

  onLoad() {
    if (this.connectionHandlerReady) {
      return
    }
    wx.onBLEConnectionStateChange((res) => {
      if (res.deviceId !== this.data.deviceId) {
        return
      }
      if (!res.connected) {
        this.releaseHold(false)
        this.setData({
          validating: false,
          connected: false,
          serviceId: '',
          controlId: '',
          lastAction: `断开 ${stamp()}`
        })
        this.addLog('蓝牙已断开')
      }
    })
    this.connectionHandlerReady = true
  },

  onUnload() {
    this.releaseHold()
    if (this.data.deviceId) {
      wx.closeBLEConnection({ deviceId: this.data.deviceId })
    }
    wx.closeBluetoothAdapter()
  },

  onSpeedChange(event) {
    const value = event.detail.value
    this.setData({
      speedValue: value,
      speedText: (value / 100).toFixed(2)
    })
  },

  scanAndConnect() {
    if (this.data.scanning) {
      this.addLog('正在扫描')
      return
    }
    this.releaseHold()
    if (this.data.deviceId) {
      wx.closeBLEConnection({ deviceId: this.data.deviceId })
    }
    this.addLog('开始扫描 G1D-BLE-RCS')
    this.scanSeen = 0
    this.scanFound = 0
    this.setData({
      scanning: true,
      validating: false,
      connected: false,
      deviceId: '',
      deviceName: '',
      serviceId: '',
      controlId: '',
      candidateDevices: []
    })
    wx.openBluetoothAdapter({
      success: () => this.startDiscovery(),
      fail: (err) => {
        this.setData({ scanning: false })
        this.addLog(`蓝牙初始化失败 ${err.errMsg || ''}`)
      }
    })
  },

  startDiscovery() {
    if (!this.deviceFoundHandlerReady) {
      wx.onBluetoothDeviceFound((res) => {
        const devices = res.devices || []
        this.scanFound += devices.length
        const candidates = devices.filter(isG1NamedDevice)
        if (candidates.length) {
          this.mergeCandidateDevices(candidates)
        }
        for (const item of devices) {
          const name = item.name || item.localName || ''
          const uuids = item.advertisServiceUUIDs || item.advertiseServiceUUIDs || []
          if (this.scanSeen < 6) {
            this.scanSeen += 1
            this.addLog(`附近设备 ${name || '无名称'} ${uuids.length ? '有UUID' : '无UUID'}`)
          }
        }
      })
      this.deviceFoundHandlerReady = true
    }
    wx.startBluetoothDevicesDiscovery({
      allowDuplicatesKey: true,
      success: () => {
        setTimeout(() => {
          if (this.data.scanning) {
            wx.stopBluetoothDevicesDiscovery()
            this.setData({ scanning: false })
            this.addLog(`未扫描到 G1D-BLE-RCS，共发现 ${this.scanFound} 条广播`)
          }
        }, 12000)
      },
      fail: (err) => {
        this.setData({ scanning: false })
        this.addLog(`扫描失败 ${err.errMsg || ''}`)
      }
    })
  },

  mergeCandidateDevices(devices) {
    const byId = {}
    for (const item of this.data.candidateDevices) {
      byId[item.deviceId] = item
    }
    for (const item of devices) {
      const name = item.name || item.localName || ''
      byId[item.deviceId] = {
        deviceId: item.deviceId,
        name,
        RSSI: item.RSSI || item.rssi || 0
      }
    }
    const candidateDevices = Object.values(byId).sort((a, b) => {
      const ar = Number(a.RSSI || -999)
      const br = Number(b.RSSI || -999)
      return br - ar
    })
    this.setData({ candidateDevices })
  },

  connectCandidate(event) {
    const deviceId = event.currentTarget.dataset.deviceId
    const device = this.data.candidateDevices.find((item) => item.deviceId === deviceId)
    if (!device) {
      this.addLog('候选设备不存在')
      return
    }
    wx.stopBluetoothDevicesDiscovery()
    this.setData({ scanning: false })
    this.connectDevice(device)
  },

  connectDevice(device) {
    const name = device.name || device.localName || 'G1'
    this.addLog(`连接 ${name}`)
    this.setData({ scanning: false, validating: true, connected: false, deviceName: name })
    wx.createBLEConnection({
      deviceId: device.deviceId,
      success: () => {
        this.setData({
          validating: true,
          connected: false,
          scanning: false,
          deviceId: device.deviceId,
          deviceName: name
        })
        this.discoverService()
      },
      fail: (err) => {
        this.setData({ scanning: false, validating: false, connected: false })
        this.addLog(`连接失败 ${err.errMsg || ''}`)
      }
    })
  },

  discoverService() {
    wx.getBLEDeviceServices({
      deviceId: this.data.deviceId,
      success: (res) => {
        const service = (res.services || []).find((item) => item.uuid.toLowerCase() === SERVICE_UUID)
        if (!service) {
          const found = (res.services || []).slice(0, 4).map((item) => item.uuid.slice(0, 8)).join(',')
          this.addLog(`未找到 G1D 服务${found ? `，服务 ${found}` : ''}`)
          wx.closeBLEConnection({ deviceId: this.data.deviceId })
          this.setData({ validating: false, connected: false, serviceId: '', controlId: '' })
          return
        }
        this.setData({ serviceId: service.uuid })
        this.discoverCharacteristics(service.uuid)
      },
      fail: (err) => {
        this.setData({ validating: false, connected: false })
        this.addLog(`读取服务失败 ${err.errMsg || ''}`)
      }
    })
  },

  discoverCharacteristics(serviceId) {
    wx.getBLEDeviceCharacteristics({
      deviceId: this.data.deviceId,
      serviceId,
      success: (res) => {
        const control = (res.characteristics || []).find((item) => item.uuid.toLowerCase() === CONTROL_UUID)
        if (!control) {
          this.addLog('未找到控制特征')
          wx.closeBLEConnection({ deviceId: this.data.deviceId })
          this.setData({ validating: false, connected: false, controlId: '' })
          return
        }
        this.setData({ validating: false, connected: true, controlId: control.uuid })
        this.addLog('BLE 已就绪')
      },
      fail: (err) => {
        this.setData({ validating: false, connected: false })
        this.addLog(`读取特征失败 ${err.errMsg || ''}`)
      }
    })
  },

  holdForward() { this.startHold('forward', '前进') },
  holdBack() { this.startHold('back', '后退') },
  holdLeft() { this.startHold('turn_left', '左转') },
  holdRight() { this.startHold('turn_right', '右转') },
  holdUp() { this.startHold('column_up', '升') },
  holdDown() { this.startHold('column_down', '降') },

  startHold(action, label) {
    if (!this.data.connected || !this.data.controlId) {
      this.addLog('请先连接蓝牙')
      return
    }
    this.releaseHold(false)
    this.holdingAction = action
    this.setData({ activeAction: action, lastAction: `${label} ${stamp()}` })
    this.writeCommand(`H ${action} ${this.data.speedText}`)
    this.heartbeatTimer = setInterval(() => {
      this.writeCommand(`H ${action} ${this.data.speedText}`, false)
      this.setData({ lastAction: `${label} ${stamp()}` })
    }, HEARTBEAT_MS)
  },

  releaseHold(sendStop = true) {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
    this.holdingAction = ''
    this.setData({ activeAction: '' })
    if (sendStop && this.data.connected && this.data.controlId) {
      this.writeCommand('S')
    }
  },

  sendStop() {
    this.releaseHold(false)
    this.setData({ lastAction: `停止 ${stamp()}` })
    this.writeCommand('S')
  },

  writeCommand(text, log = true) {
    if (!this.data.deviceId || !this.data.serviceId || !this.data.controlId) {
      return
    }
    wx.writeBLECharacteristicValue({
      deviceId: this.data.deviceId,
      serviceId: this.data.serviceId,
      characteristicId: this.data.controlId,
      value: textToArrayBuffer(text),
      fail: (err) => this.addLog(`发送失败 ${err.errMsg || ''}`)
    })
    if (log) {
      this.addLog(`发送 ${text}`)
    }
  },

  addLog(text) {
    const logs = [`${stamp()} ${text}`, ...this.data.logs].slice(0, 8)
    this.setData({ logs })
  }
})
