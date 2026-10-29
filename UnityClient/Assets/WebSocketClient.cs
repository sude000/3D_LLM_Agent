using System;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using UnityEngine;
using NativeWebSocket;

[System.Serializable]
public class SingleCommand
{
    public string action;
    public string target;
    public float[] pos;
}

[System.Serializable]
public class MultiCommandData
{
    public List<SingleCommand> actions = new List<SingleCommand>();
}

[System.Serializable]
public class ObjectState
{
    public string name;
    public string color; // Dinamik algýlanan renk
    public float[] position;
}

[System.Serializable]
public class SceneStateData
{
    public string type = "scene_state";
    public List<ObjectState> objects = new List<ObjectState>();
}

public class WebSocketClient : MonoBehaviour
{
    private WebSocket websocket;

    [Header("Etkileþimli Nesneler Listesi")]
    public List<GameObject> interactiveObjects = new List<GameObject>();

    private bool isReconnecting = false;
    private bool isQuitting = false;

    async void Start()
    {
        await ConnectToWebSocket();
    }

    async Task ConnectToWebSocket()
    {
        if (isQuitting) return;

        websocket = new WebSocket("ws://localhost:8080");

        websocket.OnOpen += () =>
        {
            Debug.Log("[Unity] WebSocket Baðlantýsý Baþarýlý!");
            isReconnecting = false;
            SendSceneStateToPython();
        };

        websocket.OnError += (e) =>
        {
            Debug.LogError("[Unity] WebSocket Hatasý: " + e);
        };

        websocket.OnClose += (e) =>
        {
            if (!isQuitting)
            {
                Debug.LogWarning("[Unity] WebSocket Baðlantýsý Koptu. Yeniden baðlanýlmaya çalýþýlýyor...");
                TryReconnect();
            }
        };

        websocket.OnMessage += (bytes) =>
        {
            var message = Encoding.UTF8.GetString(bytes);
            Debug.Log("[Unity] Gelen Mesaj: " + message);
            ProcessMessage(message);
        };

        await websocket.Connect();
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        websocket?.DispatchMessageQueue();
#endif
    }

    public string GetSceneStateJson()
    {
        SceneStateData stateData = new SceneStateData();

        foreach (GameObject obj in interactiveObjects)
        {
            if (obj != null)
            {
                ObjectState state = new ObjectState();
                state.name = obj.name.ToLower();

                // --- DÝNAMÝK RENK ALGILAMA BÖLÜMÜ ---
                state.color = DetectObjectColor(obj);

                Vector3 pos = obj.transform.position;
                state.position = new float[] { (float)Math.Round(pos.x, 2), (float)Math.Round(pos.y, 2), (float)Math.Round(pos.z, 2) };
                stateData.objects.Add(state);
            }
        }

        return JsonUtility.ToJson(stateData);
    }

    /// <summary>
    /// Nesnenin Renderer bileþenindeki Materyal rengini analiz eder ve Türkçe renk adýný döner.
    /// </summary>
    private string DetectObjectColor(GameObject obj)
    {
        Renderer renderer = obj.GetComponent<Renderer>();
        if (renderer == null || renderer.material == null)
            return "bilinmiyor";

        // Varsa materyal adýndan renk okuma (Örn: "RedMaterial" -> kýrmýzý)
        string matName = renderer.material.name.ToLower();
        if (matName.Contains("red") || matName.Contains("kýrmýzý")) return "kýrmýzý";
        if (matName.Contains("blue") || matName.Contains("mavi")) return "mavi";
        if (matName.Contains("green") || matName.Contains("yeþil")) return "yeþil";
        if (matName.Contains("yellow") || matName.Contains("sarý")) return "sarý";
        if (matName.Contains("black") || matName.Contains("siyah")) return "siyah";
        if (matName.Contains("white") || matName.Contains("beyaz")) return "beyaz";

        // Materyal adýnda yoksa RGB deðerlerine bakarak renk tahmini
        Color c = renderer.material.color;

        // Basit RGB Algýlama Mantýðý
        if (c.r > 0.6f && c.g < 0.4f && c.b < 0.4f) return "kýrmýzý";
        if (c.b > 0.6f && c.r < 0.4f && c.g < 0.4f) return "mavi";
        if (c.g > 0.6f && c.r < 0.4f && c.b < 0.4f) return "yeþil";
        if (c.r > 0.6f && c.g > 0.6f && c.b < 0.4f) return "sarý";
        if (c.r < 0.2f && c.g < 0.2f && c.b < 0.2f) return "siyah";
        if (c.r > 0.8f && c.g > 0.8f && c.b > 0.8f) return "beyaz";

        return "renkli";
    }

    public async void SendSceneStateToPython()
    {
        string jsonState = GetSceneStateJson();
        await SendMessageToPython(jsonState);
        Debug.Log("[Unity] Güncel Sahne Durumu Python'a Gönderildi: " + jsonState);
    }

    private void ProcessMessage(string jsonMessage)
    {
        try
        {
            MultiCommandData multiCommand = JsonUtility.FromJson<MultiCommandData>(jsonMessage);

            if (multiCommand != null && multiCommand.actions != null && multiCommand.actions.Count > 0)
            {
                foreach (SingleCommand command in multiCommand.actions)
                {
                    if (command.action == "move")
                    {
                        GameObject targetObj = interactiveObjects.Find(o => o.name.ToLower() == command.target.ToLower());

                        if (targetObj != null && command.pos.Length == 3)
                        {
                            Vector3 offset = new Vector3(command.pos[0], command.pos[1], command.pos[2]);
                            targetObj.transform.position += offset;

                            Vector3 updatedPos = targetObj.transform.position;
                            Debug.Log($"[Unity] {command.target} objesi {offset} miktar kaydýrýldý. Yeni Güncel Konum: {updatedPos}");
                        }
                        else
                        {
                            Debug.LogWarning($"[Unity] Target obje bulunamadý veya pos hatalý: {command.target}");
                        }
                    }
                }

                // Tüm eylemler tamamlandýktan sonra güncel sahne durumunu Python'a bildir
                SendSceneStateToPython();
            }
        }
        catch (Exception ex)
        {
            Debug.LogError("[Unity] Mesaj iþlenirken hata oluþtu: " + ex.Message);
        }
    }

    public async Task SendMessageToPython(string message)
    {
        if (websocket != null && websocket.State == WebSocketState.Open)
        {
            await websocket.SendText(message);
        }
    }

    private async void TryReconnect()
    {
        if (isReconnecting || isQuitting) return;
        isReconnecting = true;

        while (!isQuitting && (websocket == null || websocket.State != WebSocketState.Open))
        {
            Debug.Log("[Unity] 3 saniye sonra tekrar baðlanýlacak...");
            await Task.Delay(3000);
            if (isQuitting) break;
            await ConnectToWebSocket();
        }
    }

    private async Task CloseWebSocket()
    {
        isQuitting = true;
        if (websocket != null)
        {
            if (websocket.State == WebSocketState.Open)
            {
                await websocket.Close();
            }
            websocket = null;
        }
    }

    private async void OnDisable() => await CloseWebSocket();
    private async void OnApplicationQuit() => await CloseWebSocket();
}