using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

public readonly record struct GameDef(ulong Id, string Name);

public readonly record struct NcaEntry(ulong TitleId, bool IsProgram, bool IsPatch, bool IsPublicData);

public sealed record ClassifyResult(
    ulong BaseTitleId,
    string? GameName,
    int BaseIndex,
    int UpdateIndex,
    IReadOnlyList<int> DlcIndices,
    string? Error);

public static class ZaImportLogic
{
    // The games set up from their packages (update: +0x800, DLC: +0x1000 + n); the first one found among them is taken
    public static readonly GameDef[] SupportedGames =
    {
        new(0x01008F6008C5E000, "Pokémon Violet"),
        new(0x0100A3D008C5C000, "Pokémon Scarlet"),
        new(0x0100F43008C44000, "Pokémon Legends Z-A"),
        new(0x01001F5010DFA000, "Pokémon Legends Arceus"),
    };

    public static ClassifyResult Classify(IReadOnlyList<NcaEntry> entries, bool aocOnly)
    {
        if (entries.Count == 0)
            return new ClassifyResult(0, null, -1, -1, Array.Empty<int>(), "no game packages found");

        ulong zaBase = SupportedGames.Select(g => g.Id).FirstOrDefault(id => entries.Any(e => e.TitleId == id));
        if (zaBase == 0 && aocOnly)
        {
            zaBase = SupportedGames
                .Select(g => g.Id)
                .FirstOrDefault(id => entries.Any(e => e.IsPublicData && (e.TitleId & ~0x1FFFUL) == id));
        }

        if (zaBase == 0)
        {
            string names = string.Join(" or ", SupportedGames.Select(g => g.Name));
            string err = aocOnly
                ? $"no DLC packages for {names} found in these files"
                : $"the base game of {names} isn't among these files";
            return new ClassifyResult(0, null, -1, -1, Array.Empty<int>(), err);
        }

        string gameName = SupportedGames.First(g => g.Id == zaBase).Name;
        int baseIdx = -1, updIdx = -1;
        var dlcByTitle = new Dictionary<ulong, int>();
        var dlcOrder = new List<ulong>();

        for (int i = 0; i < entries.Count; i++)
        {
            var e = entries[i];
            if ((e.TitleId & ~0x1FFFUL) != zaBase) continue;
            if (e.IsProgram)
            {
                if (e.IsPatch) updIdx = i;
                else if (e.TitleId == zaBase) baseIdx = i;
            }
            else if (e.IsPublicData)
            {
                if (!dlcByTitle.ContainsKey(e.TitleId)) dlcOrder.Add(e.TitleId);
                dlcByTitle[e.TitleId] = i;
            }
        }

        if (!aocOnly && baseIdx < 0)
        {
            string names = string.Join(" or ", SupportedGames.Select(g => g.Name));
            return new ClassifyResult(zaBase, gameName, -1, updIdx, Array.Empty<int>(),
                $"the base game of {names} isn't among these files");
        }

        var dlcIndices = dlcOrder.Select(tid => dlcByTitle[tid]).ToList();
        if (aocOnly && dlcIndices.Count == 0)
        {
            return new ClassifyResult(zaBase, gameName, baseIdx, updIdx, Array.Empty<int>(),
                $"no {gameName} DLC packages found in these files");
        }

        return new ClassifyResult(zaBase, gameName, baseIdx, updIdx, dlcIndices, null);
    }

    public static string ResolveAocOutputPath(string outDir, ulong titleId)
    {
        string aocRoot = Path.GetFullPath(Path.Combine(outDir, "aoc"));
        string target = Path.GetFullPath(Path.Combine(aocRoot, $"{titleId:x16}.bin"));
        string prefix = aocRoot.EndsWith(Path.DirectorySeparatorChar)
            ? aocRoot
            : aocRoot + Path.DirectorySeparatorChar;
        if (!target.StartsWith(prefix, StringComparison.Ordinal))
            throw new InvalidOperationException($"Refusing to write AOC outside {aocRoot}");
        return target;
    }
}
