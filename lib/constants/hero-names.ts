import heroReferenceAssets from "@/config/hero_reference_assets.json";

export type HeroReferenceAsset = {
  id: string;
  base_name: string;
  is_lord: boolean;
  files: string[];
};

export const HERO_REFERENCE_ASSETS: HeroReferenceAsset[] =
  heroReferenceAssets as HeroReferenceAsset[];

export const HERO_NAMES = HERO_REFERENCE_ASSETS.map((entry) => entry.id);

export const HERO_NAME_SET = new Set(HERO_NAMES);
